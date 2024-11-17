use std::io::Cursor;
use std::mem;
use std::mem::{align_of, size_of, size_of_val};
use std::borrow::Cow;
use std::default::Default;
use std::ffi;
use std::os::raw::c_char;
use cgmath::{perspective, InnerSpace, Matrix4 as Mat4, Point3, Rad, SquareMatrix};
use cgmath::Vector3 as Vec3;

use ash::vk;
use ash::khr;
use windows_sys::Win32::Foundation::HWND;
use crate::{example_model, win};

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}

/// Helper function for submitting command buffers. Immediately waits for the fence before the command buffer
/// is executed. That way we can delay the waiting for the fences by 1 frame which is good for performance.
/// Make sure to create the fence in a signaled state on the first use.
pub fn record_submit_commandbuffer<F: FnOnce(&ash::Device, vk::CommandBuffer)>(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    unsafe {
        device
            .wait_for_fences(&[command_buffer_reuse_fence], true, u64::MAX)
            .expect("Wait for fence failed.");
        device
            .reset_fences(&[command_buffer_reuse_fence])
            .expect("Reset fences failed.");

        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");
        f(device, command_buffer);
        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(submit_queue, &[submit_info], command_buffer_reuse_fence)
            .expect("queue submit failed.");
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void, ) -> vk::Bool32
{
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    vk::FALSE
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32>
{
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) as u32 & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}

// todo: separate further into separate files and add specific functions
// todo: that make it easier to add things without restructuring code
// todo: after first making creation of (Base, Target, Sync, Process) clear oneliners
// todo: and then looking to see what is needed afterwards to run things
// todo: and then add functions to add stuff to it easily in main, ex. shaders, vertex models, etc.
pub struct Base {
    pub device: ash::Device,
    pub pdevice: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub surface_loader: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub swapchain_loader: khr::swapchain::Device,
    pub queue_family_index: u32
}

pub struct Target<'a> {
    pub surface_resolution: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_memory: vk::DeviceMemory,
    pub renderpass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub viewports: [vk::Viewport; 1],
    pub scissors: [vk::Rect2D; 1],
    pub viewport_state_info: vk::PipelineViewportStateCreateInfo<'a>
}

pub struct Process {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipelines: Vec<vk::Pipeline>,
    
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub index_buffer_data: Vec<u16>,
    
    pub vertex_shader_module: vk::ShaderModule,
    pub fragment_shader_module: vk::ShaderModule,
    
    pub uniform_buffer: vk::Buffer,
    pub uniform_buffer_memory: vk::DeviceMemory,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub camera: Camera,
}

pub struct Sync {
    pub present_queue: vk::Queue,
    pub frames_in_flight: usize,
    pub frame_data: Vec<FrameData>,
    pub pool: vk::CommandPool,
    pub setup_commands_reuse_fence: vk::Fence
}

pub struct FrameData {
    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,
    pub fence: vk::Fence,
    pub command_buffer: vk::CommandBuffer,
}

pub unsafe fn vk_create_base(hwnd: HWND) -> Base {
    let entry = ash::Entry::linked();
    let app_name = ffi::CStr::from_bytes_with_nul_unchecked(b"VulkanTriangle\0");

    let layer_names = [ffi::CStr::from_bytes_with_nul_unchecked(
        b"VK_LAYER_KHRONOS_validation\0",
    )];
    let layers_names_raw: Vec<*const c_char> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let appinfo = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(0)
        .engine_name(app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    // todo: win.rs should be main() entrypoint, and it calls into vulkan code and application code
    // instead of the other way round (the way it is now)
    let (instance, surface) = win::setup_windows_window(hwnd, &entry, &layers_names_raw, &appinfo, create_flags);

    let surface_loader = khr::surface::Instance::new(&entry, &instance);

    let (pdevice, queue_family_index) = select_physical_device(&instance, &surface_loader, surface)
        .expect("Failed to find a suitable GPU!");
    let queue_family_index = queue_family_index as u32;
    let device_extension_names_raw = [
        khr::swapchain::NAME.as_ptr()
    ];
    let features = vk::PhysicalDeviceFeatures {
        shader_clip_distance: 1,
        ..Default::default()
    };
    let priorities = [1.0];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&device_extension_names_raw)
        .enabled_features(&features);
    let device: ash::Device = instance
        .create_device(pdevice, &device_create_info, None)
        .unwrap();
    let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

    let surface_format = surface_loader
        .get_physical_device_surface_formats(pdevice, surface)
        .unwrap()[0];
    let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

    // region DEBUG TOOLS
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
    let debug_call_back = debug_utils_loader
        .create_debug_utils_messenger(&debug_info, None)
        .unwrap();
    // endregion DEBUG TOOLS

    Base {
        device,
        pdevice,
        device_memory_properties,
        surface_loader,
        surface,
        surface_format,
        swapchain_loader,
        queue_family_index,
    }
}

pub unsafe fn vk_create_sync(base: &Base, target: &Target) -> Sync {
    let present_queue = base.device.get_device_queue(base.queue_family_index, 0);

    let fence_create_info =
        vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    let setup_commands_reuse_fence = base.device
        .create_fence(&fence_create_info, None)
        .expect("Create fence failed.");

    let pool_create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(base.queue_family_index);

    let pool = base.device.create_command_pool(&pool_create_info, None).unwrap();

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(2)
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let setup_command_buffers = base.device
        .allocate_command_buffers(&command_buffer_allocate_info)
        .unwrap();
    let setup_command_buffer = setup_command_buffers[0];

    record_submit_commandbuffer(
        &base.device,
        setup_command_buffer,
        setup_commands_reuse_fence,
        present_queue,
        &[],
        &[],
        &[],
        |device, setup_command_buffer| {
            let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                // todo: there seems to be a dependency on the size of the depth image
                // todo: but this is never updated (???) does this cause resize issues (?)
                .image(target.depth_image)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .layer_count(1)
                        .level_count(1),
                );

            device.cmd_pipeline_barrier(
                setup_command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[layout_transition_barriers],
            );
        },
    );

    let frames_in_flight = 2;
    let mut frame_data: Vec<FrameData> = Vec::with_capacity(frames_in_flight);

    // Allocate command buffers for each frame
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(frames_in_flight as u32)
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let command_buffers = base.device
        .allocate_command_buffers(&command_buffer_allocate_info)
        .unwrap();

    for i in 0..frames_in_flight {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        let present_complete_semaphore = base.device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap();
        let rendering_complete_semaphore = base.device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap();
        let fence = base.device
            .create_fence(&fence_create_info, None)
            .unwrap();

        frame_data.push(FrameData {
            present_complete_semaphore,
            rendering_complete_semaphore,
            fence,
            command_buffer: command_buffers[i],
        });
    }

    let sync = Sync {
        present_queue,
        pool,
        frames_in_flight,
        frame_data,
        setup_commands_reuse_fence,
    };
    sync
}

/// If there was already a target with a swapchain, the old swapchain must be provided
/// to avoid a panic when creating the new swapchain
pub unsafe fn setup_swapchain(base: &Base, old_swapchain: Option<vk::SwapchainKHR>) -> Target {
    // get the current resolution of the surface from base
    let surface_capabilities = base.surface_loader
        .get_physical_device_surface_capabilities(base.pdevice, base.surface)
        .unwrap();
    let surface_resolution = surface_capabilities.current_extent;

    let depth_image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::D16_UNORM)
        .extent(surface_resolution.into())
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let depth_image = base.device.create_image(&depth_image_create_info, None).unwrap();
    let depth_image_memory_req = base.device.get_image_memory_requirements(depth_image);
    let depth_image_memory_index = find_memorytype_index(
        &depth_image_memory_req,
        &base.device_memory_properties,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
        .expect("Unable to find suitable memory index for depth image.");

    let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(depth_image_memory_req.size)
        .memory_type_index(depth_image_memory_index);

    let depth_image_memory = base.device
        .allocate_memory(&depth_image_allocate_info, None)
        .unwrap();

    base.device
        .bind_image_memory(depth_image, depth_image_memory, 0)
        .expect("Unable to bind depth image memory");

    let depth_image_view_info = vk::ImageViewCreateInfo::default()
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .level_count(1)
                .layer_count(1),
        )
        .image(depth_image)
        .format(depth_image_create_info.format)
        .view_type(vk::ImageViewType::TYPE_2D);

    let depth_image_view = base.device
        .create_image_view(&depth_image_view_info, None)
        .unwrap();

    let present_modes = base.surface_loader
        .get_physical_device_surface_present_modes(base.pdevice, base.surface)
        .unwrap();
    let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    };

    // Determine optimal image count
    let mut desired_image_count = surface_capabilities.min_image_count + 1;
    if surface_capabilities.max_image_count > 0 {
        desired_image_count = desired_image_count.min(surface_capabilities.max_image_count);
    }

    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(base.surface)
        .min_image_count(desired_image_count)
        .image_color_space(base.surface_format.color_space)
        .image_format(base.surface_format.format)
        .image_extent(surface_resolution)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1);
    
    // if there was already a swapchain, provide this info
    if (old_swapchain.is_some()) {
        swapchain_create_info = swapchain_create_info.old_swapchain(old_swapchain.unwrap());
    }

    let swapchain = base.swapchain_loader
        .create_swapchain(&swapchain_create_info, None)
        .unwrap();

    let present_images = base.swapchain_loader.get_swapchain_images(swapchain).unwrap();
    let present_image_views: Vec<vk::ImageView> = present_images
        .iter()
        .map(|&image| {
            let create_view_info = vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(base.surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image);
            base.device.create_image_view(&create_view_info, None).unwrap()
        })
        .collect();

    let renderpass_attachments = [
        vk::AttachmentDescription {
            format: base.surface_format.format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        },
        vk::AttachmentDescription {
            format: vk::Format::D16_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        },
    ];
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_attachment_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..Default::default()
    }];

    let subpass = vk::SubpassDescription::default()
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

    let renderpass_create_info = vk::RenderPassCreateInfo::default()
        .attachments(&renderpass_attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&dependencies);

    let renderpass = base.device
        .create_render_pass(&renderpass_create_info, None)
        .unwrap();

    let framebuffers: Vec<vk::Framebuffer> = present_image_views
        .iter()
        .map(|&present_image_view| {
            let framebuffer_attachments = [present_image_view, depth_image_view];
            let frame_buffer_create_info = vk::FramebufferCreateInfo::default()
                .render_pass(renderpass)
                .attachments(&framebuffer_attachments)
                .width(surface_resolution.width)
                .height(surface_resolution.height)
                .layers(1);

            base.device
                .create_framebuffer(&frame_buffer_create_info, None)
                .unwrap()
        })
        .collect();

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: surface_resolution.width as f32,
        height: surface_resolution.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: surface_resolution,
    }];
    let viewport_state_info = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: Default::default(),
        viewport_count: 1,
        p_viewports: viewports.as_ptr(),
        scissor_count: 1,
        p_scissors: scissors.as_ptr(),
        _marker: Default::default(),
    };

    // Destroy old swapchain after new one is created
    if old_swapchain.is_some() {
        base.swapchain_loader.destroy_swapchain(old_swapchain.unwrap(), None);
    }

    // create a new Target to replace the old one
    Target {
        swapchain,
        present_images,
        present_image_views,
        surface_resolution,
        depth_image,
        depth_image_memory,
        depth_image_view,
        renderpass,
        framebuffers,
        viewports,
        scissors,
        viewport_state_info,
    }
}

fn select_physical_device(
    instance: &ash::Instance,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Option<(vk::PhysicalDevice, u32)> {
    unsafe {
        let pdevices = instance.enumerate_physical_devices().ok()?;

        // Score and sort physical devices
        let mut device_scores: Vec<(vk::PhysicalDevice, i32)> = pdevices
            .iter()
            .map(|&pdevice| {
                let props = instance.get_physical_device_properties(pdevice);
                let score = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 50,
                    vk::PhysicalDeviceType::CPU => 10,
                    _ => 0,
                };
                (pdevice, score)
            })
            .collect();

        device_scores.sort_by(|a, b| b.1.cmp(&a.1));

        // Find the first suitable device
        for (pdevice, _) in device_scores {
            if let Some(queue_family_index) = instance
                .get_physical_device_queue_family_properties(pdevice)
                .iter()
                .enumerate()
                .find(|(index, info)| {
                    info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && surface_loader
                        .get_physical_device_surface_support(
                            pdevice,
                            *index as u32,
                            surface,
                        )
                        .unwrap_or(false)
                })
                .map(|(index, _)| index as u32)
            {
                return Some((pdevice, queue_family_index));
            }
        }
        None
    }
}

// PROCESS STUFF: PIPELINE, SHADERS, CAMERA, ...

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject { // todo: why is this called this, and not Transform or something?
    model: Mat4<f32>,
    view: Mat4<f32>,
    proj: Mat4<f32>,
}

#[derive(Debug)]
pub struct Camera {
    position: Point3<f32>,
    front: Vec3<f32>,
    up: Vec3<f32>,
    right: Vec3<f32>,
    world_up: Vec3<f32>,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    pub fn new(position: Point3<f32>) -> Self {
        let mut camera = Camera {
            position,
            front: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            world_up: Vec3::new(0.0, 1.0, 0.0),
            yaw: -90.0,
            pitch: 0.0,
        };
        camera.update_camera_vectors();
        camera
    }

    pub fn get_view_matrix(&self) -> Mat4<f32> {
        Mat4::look_at_rh(self.position, self.position + self.front, self.up)
    }

    pub fn move_forward(&mut self, delta: f32) {
        self.position += self.front * delta;
    }

    pub fn move_backward(&mut self, delta: f32) {
        self.position -= self.front * delta;
    }

    pub fn move_right(&mut self, delta: f32) {
        self.position += self.right * delta;
    }

    pub fn move_left(&mut self, delta: f32) {
        self.position -= self.right * delta;
    }

    pub fn rotate(&mut self, yaw_delta: f32, pitch_delta: f32) {
        self.yaw += yaw_delta; // right left
        self.pitch += pitch_delta; // up down

        // Constrain pitch
        self.pitch = self.pitch.clamp(-89.0, 89.0);

        self.update_camera_vectors();
    }

    fn update_camera_vectors(&mut self) {
        let (yaw_rad, pitch_rad) = (self.yaw.to_radians(), self.pitch.to_radians());

        self.front = Vec3::new(
            yaw_rad.cos() * pitch_rad.cos(),
            pitch_rad.sin(),
            yaw_rad.sin() * pitch_rad.cos(),
        ).normalize();

        self.right = self.front.cross(self.world_up).normalize();
        self.up = self.right.cross(self.front).normalize();
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub color: [f32; 4],
}

unsafe fn create_uniform_buffer(base: &Base) -> (
    vk::Buffer,
    vk::DeviceMemory,
    vk::DescriptorPool,
    vk::DescriptorSetLayout,
    Vec<vk::DescriptorSet>
) {
    // 1. Create uniform buffer
    let buffer_size = std::mem::size_of::<UniformBufferObject>() as u64;

    let uniform_buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let uniform_buffer = base.device.create_buffer(&uniform_buffer_info, None).unwrap();

    let mem_requirements = base.device.get_buffer_memory_requirements(uniform_buffer);
    let memory_type = find_memorytype_index(
        &mem_requirements,
        &base.device_memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ).unwrap();

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let uniform_buffer_memory = base.device.allocate_memory(&alloc_info, None).unwrap();
    base.device.bind_buffer_memory(uniform_buffer, uniform_buffer_memory, 0).unwrap();

    // 2. Create descriptor set layout
    let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);
    let bindings = &[ubo_layout_binding];
    let layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(bindings);
    let descriptor_set_layout = base.device
        .create_descriptor_set_layout(&layout_create_info, None)
        .unwrap();

    // 3. Create descriptor pool
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1);
    let pool_sizes = &[pool_size];
    let pool_create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(pool_sizes)
        .max_sets(1);

    let descriptor_pool = base.device
        .create_descriptor_pool(&pool_create_info, None)
        .unwrap();

    // 4. Create descriptor sets
    let layouts = [descriptor_set_layout];
    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_sets = base.device
        .allocate_descriptor_sets(&descriptor_set_allocate_info)
        .unwrap();

    // 5. Update descriptor set
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(uniform_buffer)
        .offset(0)
        .range(std::mem::size_of::<UniformBufferObject>() as u64);
    let info_array = &[buffer_info];
    let descriptor_write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_sets[0])
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(info_array);

    base.device.update_descriptor_sets(&[descriptor_write], &[]);

    (
        uniform_buffer,
        uniform_buffer_memory,
        descriptor_pool,
        descriptor_set_layout,
        descriptor_sets
    )
}


impl Process {
    pub fn update_uniform_buffer(&self, base: &Base, target: &Target) {
        let ubo = UniformBufferObject {
            model: Mat4::identity(),
            view: self.camera.get_view_matrix(),
            proj: perspective(
                Rad(45.0_f32.to_radians()),
                target.surface_resolution.width as f32 / target.surface_resolution.height as f32,
                0.1,
                100.0,
            ),
        };

        unsafe {
            let data_ptr = base.device.map_memory(
                self.uniform_buffer_memory,
                0,
                std::mem::size_of::<UniformBufferObject>() as u64,
                vk::MemoryMapFlags::empty(),
            ).unwrap();

            std::ptr::copy_nonoverlapping(
                &ubo as *const UniformBufferObject,
                data_ptr as *mut UniformBufferObject,
                1,
            );

            base.device.unmap_memory(self.uniform_buffer_memory);
        }
    }
}

pub unsafe fn vk_create_process(base: &Base, target: &Target) -> Process {
    
    // todo: camera position + move camera
    // todo: 5-6% cpu usage
    // todo: function to add models from main (vertices + indices)
    // todo: give a position to a model apart from vertices
    // todo: how to structure model with pipeline?
        // model + shader?
        // can we reuse shader between multiple models?
        // can we reuse pipeline between models?

    let indices: Vec<u16> = example_model::get_indices();

    // region INDEX_BUFFER
    let index_buffer_info = vk::BufferCreateInfo::default()
        .size((indices.len() * size_of::<u16>()) as u64)
        .usage(vk::BufferUsageFlags::INDEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let index_buffer = base.device.create_buffer(&index_buffer_info, None).unwrap();
    let index_buffer_memory_req = base.device.get_buffer_memory_requirements(index_buffer);
    let index_buffer_memory_index = find_memorytype_index(
        &index_buffer_memory_req,
        &base.device_memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ).expect("Unable to find suitable memorytype for the index buffer.");
    let index_allocate_info = vk::MemoryAllocateInfo {
        allocation_size: index_buffer_memory_req.size,
        memory_type_index: index_buffer_memory_index,
        ..Default::default()
    };
    let index_buffer_memory = base
        .device
        .allocate_memory(&index_allocate_info, None)
        .unwrap();
    let index_ptr = base
        .device
        .map_memory(
            index_buffer_memory,
            0,
            index_buffer_memory_req.size,
            vk::MemoryMapFlags::empty(),
        )
        .unwrap();
    let mut index_slice = ash::util::Align::new(
        index_ptr,
        align_of::<u16>() as u64, // todo: u16 here depends on indices type
        index_buffer_memory_req.size,
    );
    index_slice.copy_from_slice(&indices);
    base.device.unmap_memory(index_buffer_memory);
    base.device
        .bind_buffer_memory(index_buffer, index_buffer_memory, 0)
        .unwrap();
    // endregion INDEX_BUFFER

    let vertices: Vec<Vertex> = example_model::get_vertices();
   
    // region VERTEX_BUFFER
    let vertex_buffer_info = vk::BufferCreateInfo {
        size: (vertices.len() * size_of::<Vertex>()) as u64,
        usage: vk::BufferUsageFlags::VERTEX_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let vertex_buffer = base
        .device
        .create_buffer(&vertex_buffer_info, None)
        .unwrap();
    let vertex_buffer_memory_req = base
        .device
        .get_buffer_memory_requirements(vertex_buffer);
    let vertex_buffer_memory_index = find_memorytype_index(
        &vertex_buffer_memory_req,
        &base.device_memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
        .expect("Unable to find suitable memorytype for the vertex buffer.");
    let vertex_buffer_allocate_info = vk::MemoryAllocateInfo {
        allocation_size: vertex_buffer_memory_req.size,
        memory_type_index: vertex_buffer_memory_index,
        ..Default::default()
    };
    let vertex_buffer_memory = base
        .device
        .allocate_memory(&vertex_buffer_allocate_info, None)
        .unwrap();
    let vert_ptr = base
        .device
        .map_memory(
            vertex_buffer_memory,
            0,
            vertex_buffer_memory_req.size,
            vk::MemoryMapFlags::empty(),
        )
        .unwrap();
    let mut vert_align = ash::util::Align::new(
        vert_ptr,
        align_of::<Vertex>() as u64,
        vertex_buffer_memory_req.size,
    );
    vert_align.copy_from_slice(&vertices);
    base.device.unmap_memory(vertex_buffer_memory);
    base.device
        .bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0)
        .unwrap();
    // endregion VERTEX_BUFFER

    // region SHADERS
    let mut vertex_spv_file =
        Cursor::new(&include_bytes!("../shader/vert.spv")[..]);
    let mut frag_spv_file = Cursor::new(&include_bytes!("../shader/frag.spv")[..]);

    let vertex_code =
        ash::util::read_spv(&mut vertex_spv_file).expect("Failed to read vertex shader spv file");
    let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&vertex_code);

    let frag_code =
        ash::util::read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file");
    let frag_shader_info = vk::ShaderModuleCreateInfo::default().code(&frag_code);

    let vertex_shader_module = base
        .device
        .create_shader_module(&vertex_shader_info, None)
        .expect("Vertex shader module error");

    let fragment_shader_module = base
        .device
        .create_shader_module(&frag_shader_info, None)
        .expect("Fragment shader module error");

    // UNIFORM BUFFER
    let (
        uniform_buffer,
        uniform_buffer_memory,
        descriptor_pool,
        descriptor_set_layout,
        descriptor_sets
    ) = create_uniform_buffer(base);
    // camera for uniform
    let camera = Camera::new(Point3 {x: 0.0, y: 0.0, z: 0.0});
    
    // PIPELINE LAYOUT
    let layouts = &[descriptor_set_layout];
    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(layouts);
    let pipeline_layout = base
        .device
        .create_pipeline_layout(&pipeline_layout_create_info, None)
        .unwrap();
    // PIPELINE LAYOUT

    let shader_entry_name = ffi::CStr::from_bytes_with_nul_unchecked(b"main\0");
    let shader_stage_create_infos = [
        vk::PipelineShaderStageCreateInfo {
            module: vertex_shader_module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            module: fragment_shader_module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];
    // endregion SHADERS

    let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let vertex_input_attribute_descriptions = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: offset_of!(Vertex, pos) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: offset_of!(Vertex, color) as u32,
        },
    ];

    let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
        .vertex_binding_descriptions(&vertex_input_binding_descriptions);
    let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

    let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        ..Default::default()
    };
    let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    let noop_stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        ..Default::default()
    };
    let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable: 1,
        depth_write_enable: 1,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        front: noop_stencil_state,
        back: noop_stencil_state,
        max_depth_bounds: 1.0,
        ..Default::default()
    };
    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
        blend_enable: 0,
        src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ZERO,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    }];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op(vk::LogicOp::CLEAR)
        .attachments(&color_blend_attachment_states);

    let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_info =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stage_create_infos)
        .vertex_input_state(&vertex_input_state_info)
        .input_assembly_state(&vertex_input_assembly_state_info)
        .rasterization_state(&rasterization_info)
        .multisample_state(&multisample_state_info)
        .depth_stencil_state(&depth_state_info)
        .color_blend_state(&color_blend_state)
        .layout(pipeline_layout)
        // dependency on TARGET
        .render_pass(target.renderpass)
        .viewport_state(&target.viewport_state_info)
        .dynamic_state(&dynamic_state_info);

    let pipelines = base
        .device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        .expect("Unable to create pipeline");

    Process {
        pipeline_layout,
        pipelines,
        vertex_buffer,
        vertex_buffer_memory,
        index_buffer,
        index_buffer_memory,
        index_buffer_data: indices,
        vertex_shader_module,
        fragment_shader_module,
        uniform_buffer,
        uniform_buffer_memory,
        descriptor_pool,
        descriptor_set_layout,
        descriptor_sets,
        camera
    }
}
