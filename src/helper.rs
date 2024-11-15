use std::{
    borrow::Cow, cell::RefCell, default::Default, error::Error, ffi, ops::Drop, os::raw::c_char,
};

use crate::win;
use ash::vk::{DeviceMemory, Extent2D, Framebuffer, Image, ImageView, PhysicalDevice, PhysicalDeviceMemoryProperties, PipelineViewportStateCreateInfo, Rect2D, RenderPass, StructureType, SurfaceFormatKHR, SurfaceKHR, SwapchainKHR, Viewport};
use ash::{ext::debug_utils, khr, khr::win32_surface, khr::{surface, swapchain}, vk, Device, Entry, Instance};
use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};
use windows_sys::Win32::Foundation::{HWND, LPARAM, POINT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::WindowsAndMessaging::{DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE, WM_QUIT};

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

// ACTUAL START OF THE FILE ----------------------------------------------------------------- //


/// Helper function for submitting command buffers. Immediately waits for the fence before the command buffer
/// is executed. That way we can delay the waiting for the fences by 1 frame which is good for performance.
/// Make sure to create the fence in a signaled state on the first use.
pub fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
    device: &Device,
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

// todo: need a render-loop function for every platform, this one is Windows (move to win.rs)
pub fn render_loop<F: FnMut() -> bool>(mut f: F) -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut msg = MSG {
            hwnd: std::ptr::null_mut(),
            message: 0,
            wParam: WPARAM::default(),
            lParam: LPARAM::default(),
            time: 0,
            pt: POINT { x: 0, y: 0 },
        };

        loop {
            // Process all pending Windows messages
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                if msg.message == WM_QUIT {
                    return Ok(());
                }
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }

            // Run a single frame of rendering
            if !f() {
                break;
            }
        }
        Ok(())
    }
}

pub fn recreate_swapchain(
    base: &mut Base,
    target: &mut Target
) -> Result<(), Box<dyn Error>> {
    unsafe {
        // wait for device to idle to do this
        base.device.device_wait_idle()?;
        // Store old swapchain for proper recreation
        let old_swapchain = target.swapchain;
        // Clean up old resources
        for framebuffer in target.framebuffers.drain(..) {
            base.device.destroy_framebuffer(framebuffer, None);
        }
        for &image_view in &target.present_image_views {
            base.device.destroy_image_view(image_view, None);
        }
        base.device.destroy_render_pass(target.renderpass, None);

        // create the swapchain stuff
        let (
            surface_resolution,
            depth_image,
            depth_image_memory,
            depth_image_view,
            swapchain,
            present_images,
            present_image_views,
            renderpass,
            framebuffers,
            viewports,
            scissors,
            viewport_state_info
        ) = setup_swapchain(
            target.surface_resolution.width,
            target.surface_resolution.height,
            target.surface,
            &target.surface_loader,
            base.pdevice,
            &base.device,
            target.surface_format,
            &base.device_memory_properties,
            &target.swapchain_loader,
            Some(old_swapchain)
        );

        // Destroy old swapchain after new one is created
        if old_swapchain != vk::SwapchainKHR::null() {
            target.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }
        // Update all the resources
        target.swapchain = swapchain;
        target.present_images = present_images;
        target.present_image_views = present_image_views;
        target.surface_resolution = surface_resolution;
        target.depth_image = depth_image;
        target.depth_image_memory = depth_image_memory;
        target.depth_image_view = depth_image_view;
        // todo: however, sync seems to depend on depth image for some reason (???) (see below)
        target.renderpass = renderpass;
        target.framebuffers = framebuffers;
        target.viewports = viewports;
        target.scissors = scissors;
        target.viewport_state_info = viewport_state_info;

        Ok(())
    }
}

pub struct Base {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub pdevice: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub debug_utils_loader: debug_utils::Instance,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,
}

pub struct Target<'a> {
    pub hwnd: HWND,
    pub surface_loader: surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,
    pub swapchain_loader: swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_memory: vk::DeviceMemory,
    pub renderpass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub viewports: [Viewport; 1],
    pub scissors: [Rect2D; 1],
    pub viewport_state_info: PipelineViewportStateCreateInfo<'a>
}

// todo: add a third struct for Process, that contains all pipeline, shaders stuff
// todo: goal is to be able to add a lot of new shaders and vertex models on the fly
// todo: without having to change anything structurally

pub struct Sync {
    pub queue_family_index: u32,
    pub present_queue: vk::Queue,
    pub pool: vk::CommandPool,
    pub frames_in_flight: usize,
    pub frame_data: Vec<FrameData>,
    pub current_frame: usize,
    pub setup_commands_reuse_fence: vk::Fence
}

pub struct FrameData {
    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,
    pub fence: vk::Fence,
    pub command_buffer: vk::CommandBuffer,
}

pub fn vk_create(window_width: u32, window_height: u32) -> Result<(Base, Target<'static>, Sync), Box<dyn Error>> {
    unsafe {
        let entry = Entry::linked();
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

        // todo: make setup function platform-independent by creating a new setup function,
        // todo: that returns Surface and Instance, (the two things that are needed down further)
        // todo: (Surface and Instance are cross-platform)
        // todo: one setup function for each platform (move this one to win.rs)
        // todo: eg. one for windows that uses HWND and create_win32_surface()
        let hwnd = win::get_hwnd(window_width as i32, window_height as i32);
        let hwnd_handle = win::WindowsHwnd::new(hwnd);
        let mut extension_names =
            ash_window::enumerate_required_extensions(RawDisplayHandle::from(hwnd_handle.display_handle().unwrap()))
                .unwrap()
                .to_vec();
        extension_names.push(debug_utils::NAME.as_ptr());

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);

        let instance: Instance = entry
            .create_instance(&create_info, None)
            .expect("Instance creation error");

        let create_info = vk::Win32SurfaceCreateInfoKHR {
            s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::Win32SurfaceCreateFlagsKHR::empty(),
            hinstance: *(GetModuleHandleW(std::ptr::null()) as *const _),
            hwnd: hwnd as isize,
            _marker: Default::default(),
        };
        let win32_instance = win32_surface::Instance::new(&entry, &instance);
        let surface = win32_instance.
            create_win32_surface(&create_info, None)
            .expect("Failed to create surface");
        // todo: end of the part that needs extracted in platform-specific setup function

        let surface_loader = surface::Instance::new(&entry, &instance);
        let (pdevice, queue_family_index) = select_physical_device(&instance, &surface_loader, surface)
            .expect("Failed to find a suitable GPU!");
        let queue_family_index = queue_family_index as u32;
        let device_extension_names_raw = [
            swapchain::NAME.as_ptr()
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

        let device: Device = instance
            .create_device(pdevice, &device_create_info, None)
            .unwrap();

        let present_queue = device.get_device_queue(queue_family_index, 0);

        let surface_format = surface_loader
            .get_physical_device_surface_formats(pdevice, surface)
            .unwrap()[0];

        let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);
        let swapchain_loader = swapchain::Device::new(&instance, &device);

        // create the swapchain stuff
        let (
        surface_resolution,
        depth_image,
        depth_image_memory,
        depth_image_view,
        swapchain,
        present_images,
        present_image_views,
        renderpass,
        framebuffers,
        viewports,
        scissors,
        viewport_state_info
        ) = setup_swapchain(
            window_width,
            window_height,
            surface,
            &surface_loader,
            pdevice,
            &device,
            surface_format,
            &device_memory_properties,
            &swapchain_loader,
            None
        );
        
        // region SYNC STUFF
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let draw_commands_reuse_fence = device
            .create_fence(&fence_create_info, None)
            .expect("Create fence failed.");
        let setup_commands_reuse_fence = device
            .create_fence(&fence_create_info, None)
            .expect("Create fence failed.");

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let pool = device.create_command_pool(&pool_create_info, None).unwrap();

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(2)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers = device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap();
        let setup_command_buffer = command_buffers[0];
        let draw_command_buffer = command_buffers[1];

        record_submit_commandbuffer(
            &device,
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
                    .image(depth_image)
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

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let present_complete_semaphore = device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap();
        let rendering_complete_semaphore = device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap();

        let frames_in_flight = 2;
        let mut frame_data = Vec::with_capacity(frames_in_flight);

        // Allocate command buffers for each frame
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(frames_in_flight as u32)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers = device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap();

        for i in 0..frames_in_flight {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let fence_create_info = vk::FenceCreateInfo::default()
                .flags(vk::FenceCreateFlags::SIGNALED);

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let fence = device
                .create_fence(&fence_create_info, None)
                .unwrap();

            frame_data.push(FrameData {
                present_complete_semaphore,
                rendering_complete_semaphore,
                fence,
                command_buffer: command_buffers[i],
            });
        }
        // endregion SYNC STUFF

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

        let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
        let debug_call_back = debug_utils_loader
            .create_debug_utils_messenger(&debug_info, None)
            .unwrap();
        // endregion DEBUG TOOLS

        let base = Base {
            entry,
            instance,
            device,
            pdevice,
            device_memory_properties,
            debug_utils_loader,
            debug_call_back,
        };

        let sync = Sync {
            queue_family_index,
            present_queue,
            pool,
            frames_in_flight,
            frame_data,
            current_frame: 0,
            setup_commands_reuse_fence,
        };

        let target = Target {
            hwnd,
            surface_loader,
            surface,
            surface_format,
            swapchain_loader,
            surface_resolution,
            swapchain,
            present_images,
            present_image_views,
            depth_image,
            depth_image_view,
            depth_image_memory,
            renderpass,
            framebuffers,
            viewports,
            scissors,
            viewport_state_info
        };

        Ok((base, target, sync))
    }
}

unsafe fn setup_swapchain<'a>(
    window_width: u32, 
    window_height: u32, 
    surface: SurfaceKHR, 
    surface_loader: &surface::Instance, 
    pdevice: PhysicalDevice, 
    device: &Device, 
    surface_format: SurfaceFormatKHR, 
    device_memory_properties: &PhysicalDeviceMemoryProperties, 
    swapchain_loader: &swapchain::Device,
    old_swapchain: Option<SwapchainKHR>
) -> (
    Extent2D, 
    Image, 
    DeviceMemory, 
    ImageView, 
    SwapchainKHR, 
    Vec<Image>, 
    Vec<ImageView>, 
    RenderPass, 
    Vec<Framebuffer>, 
    [Viewport; 1], 
    [Rect2D; 1], 
    PipelineViewportStateCreateInfo<'a>
) {
    // region DUPLICATED
    let surface_capabilities = surface_loader
        .get_physical_device_surface_capabilities(pdevice, surface)
        .unwrap();
    let surface_resolution = match surface_capabilities.current_extent.width {
        u32::MAX => vk::Extent2D {
            width: window_width,
            height: window_height,
        },
        _ => surface_capabilities.current_extent,
    };

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

    let depth_image = device.create_image(&depth_image_create_info, None).unwrap();
    let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
    let depth_image_memory_index = find_memorytype_index(
        &depth_image_memory_req,
        &device_memory_properties,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
        .expect("Unable to find suitable memory index for depth image.");

    let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(depth_image_memory_req.size)
        .memory_type_index(depth_image_memory_index);

    let depth_image_memory = device
        .allocate_memory(&depth_image_allocate_info, None)
        .unwrap();

    device
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

    let depth_image_view = device
        .create_image_view(&depth_image_view_info, None)
        .unwrap();

    let present_modes = surface_loader
        .get_physical_device_surface_present_modes(pdevice, surface)
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
        .surface(surface)
        .min_image_count(desired_image_count)
        .image_color_space(surface_format.color_space)
        .image_format(surface_format.format)
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

    let swapchain = swapchain_loader
        .create_swapchain(&swapchain_create_info, None)
        .unwrap();

    let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
    let present_image_views: Vec<vk::ImageView> = present_images
        .iter()
        .map(|&image| {
            let create_view_info = vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
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
            device.create_image_view(&create_view_info, None).unwrap()
        })
        .collect();

    let renderpass_attachments = [
        vk::AttachmentDescription {
            format: surface_format.format,
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

    let mut renderpass = device
        .create_render_pass(&renderpass_create_info, None)
        .unwrap();

    let mut framebuffers: Vec<vk::Framebuffer> = present_image_views
        .iter()
        .map(|&present_image_view| {
            let framebuffer_attachments = [present_image_view, depth_image_view];
            let frame_buffer_create_info = vk::FramebufferCreateInfo::default()
                .render_pass(renderpass)
                .attachments(&framebuffer_attachments)
                .width(surface_resolution.width)
                .height(surface_resolution.height)
                .layers(1);

            device
                .create_framebuffer(&frame_buffer_create_info, None)
                .unwrap()
        })
        .collect();

    let mut viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: surface_resolution.width as f32,
        height: surface_resolution.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let mut scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: surface_resolution,
    }];
    let mut viewport_state_info = vk::PipelineViewportStateCreateInfo {
        s_type: StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: Default::default(),
        viewport_count: 1,
        p_viewports: viewports.as_ptr(),
        scissor_count: 1,
        p_scissors: scissors.as_ptr(),
        _marker: Default::default(),
    };
    // endregion DUPLICATED
    (surface_resolution,
     depth_image,
     depth_image_memory, 
     depth_image_view,
     swapchain,
     present_images,
     present_image_views, 
     renderpass,
     framebuffers,
     viewports,
     scissors,
     viewport_state_info)
}

fn select_physical_device(
    instance: &Instance,
    surface_loader: &surface::Instance,
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

fn vk_drop(base: &mut Base, target: &mut Target, sync: &mut Sync) {
    unsafe {
        base.device.device_wait_idle().unwrap();
        // Clean up frame data
        for frame in &sync.frame_data {
            base.device.destroy_semaphore(frame.present_complete_semaphore, None);
            base.device.destroy_semaphore(frame.rendering_complete_semaphore, None);
            base.device.destroy_fence(frame.fence, None);
        }
        base.device.destroy_fence(sync.setup_commands_reuse_fence, None);
        base.device.free_memory(target.depth_image_memory, None);
        base.device.destroy_image_view(target.depth_image_view, None);
        base.device.destroy_image(target.depth_image, None);
        for &image_view in target.present_image_views.iter() {
            base.device.destroy_image_view(image_view, None);
        }
        base.device.destroy_command_pool(sync.pool, None);
        target.swapchain_loader
            .destroy_swapchain(target.swapchain, None);
        base.device.destroy_device(None);
        target.surface_loader.destroy_surface(target.surface, None);
        base.debug_utils_loader
            .destroy_debug_utils_messenger(base.debug_call_back, None);
        base.instance.destroy_instance(None);
    }
}
