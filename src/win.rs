use crate::win;
use ash::ext::debug_utils;
use ash::khr::win32_surface;
use ash::vk::{ApplicationInfo, InstanceCreateFlags, SurfaceKHR};
use ash::{vk, Entry, Instance};
use raw_window_handle::{DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowHandle, WindowsDisplayHandle};
use std::error::Error;
use std::ffi::c_void;
use std::num::NonZeroIsize;
use std::os::raw::c_char;
use std::ptr::null_mut;
use windows_sys::Win32::Foundation::{HINSTANCE, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::Foundation::{HWND, POINT};
use windows_sys::Win32::Graphics::Gdi::{ClientToScreen, HBRUSH};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_A, VK_LBUTTON, VK_R, VK_S, VK_W};
use windows_sys::Win32::{
    System::LibraryLoader::GetModuleHandleW,
    UI::WindowsAndMessaging::*,
};
use windows_sys::Win32::UI::Input::{GetRawInputData, HRAWINPUT, RAWINPUT, RAWINPUTHEADER, RID_INPUT, RIM_TYPEMOUSE};

/// Struct that takes a windows HWND and implements DisplayHandle and WindowHandle,
/// to be able to use it as window for graphics apis
pub struct WindowsHwnd {
    hwnd: NonZeroIsize,
}

impl WindowsHwnd {
    pub fn new(hwnd: HWND) -> Self {
        // Convert HWND to NonZeroIsize directly
        let hwnd_nonzero = NonZeroIsize::new(hwnd as isize)
            .expect("HWND should never be null");
        Self { hwnd: hwnd_nonzero }
    }

    pub fn get_hwnd(&self) -> HWND {
        self.hwnd.get() as HWND
    }
}

// Implement HasDisplayHandle
impl HasDisplayHandle for WindowsHwnd {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        // Create a WindowsDisplayHandle
        let handle = WindowsDisplayHandle::new();
        // Convert it to RawDisplayHandle and create a DisplayHandle
        Ok(unsafe { DisplayHandle::borrow_raw(RawDisplayHandle::Windows(handle)) })
    }
}

// Implement HasWindowHandle
impl HasWindowHandle for WindowsHwnd {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let handle = Win32WindowHandle::new(self.hwnd);
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
    }
}

/// Function that creates the windows HWND window
pub unsafe fn get_hwnd(width: i32, height: i32) -> HWND {
    // Register window class
    let hinstance = GetModuleHandleW(null_mut());
    let class_name = widestring::U16CString::from_str("MyWindowClass").unwrap();

    let wnd_class = WNDCLASSW {
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wnd_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: hinstance,
        hIcon: 0 as HICON,
        hCursor: LoadCursorW(0 as HINSTANCE, IDC_ARROW),
        hbrBackground: 0 as HBRUSH,
        lpszMenuName: std::ptr::null(),
        lpszClassName: class_name.as_ptr(),
    };
    RegisterClassW(&wnd_class);

    // Create window
    let hwnd = CreateWindowExW(
        WS_EX_LAYERED, // Extended window style for layered window
        class_name.as_ptr(),
        widestring::U16CString::from_str("My Window").unwrap().as_ptr(),
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        width,
        height,
        null_mut(),
        null_mut(),
        hinstance,
        null_mut(),
    );

    // Set layered window attributes if needed
    SetLayeredWindowAttributes(
        hwnd,
        0,    // Color key
        128,  // Alpha
        LWA_ALPHA,
    );
    hwnd
}

pub unsafe fn setup_windows_window(hwnd: HWND, entry: &Entry, layers_names_raw: &Vec<*const c_char>, appinfo: &ApplicationInfo, create_flags: InstanceCreateFlags) -> (Instance, SurfaceKHR) {
    let hwnd_handle = win::WindowsHwnd::new(hwnd);
    let mut extension_names =
        ash_window::enumerate_required_extensions(RawDisplayHandle::from(hwnd_handle.display_handle().unwrap()))
            .unwrap()
            .to_vec();
    extension_names.push(debug_utils::NAME.as_ptr());

    // toggle_cursor(false);  // Hide cursor (for first person camera)
    center_cursor(hwnd);   // Center it

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&appinfo)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names)
        .flags(create_flags);

    // todo: this seems to cause 2-3s delay at startup
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
    (instance, surface)
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

            if !f() {
                break;
            }
        }
        Ok(())
    }
}

// Window callback for events to keep the window responsive
unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_DESTROY => {
            PostQuitMessage(0);
            0
        },
        WM_INPUT => {
            let mut data_size: u32 = 0;

            // First get the size of the input data
            unsafe {
                GetRawInputData(
                    lparam as HRAWINPUT,
                    RID_INPUT,
                    null_mut(),
                    &mut data_size,
                    std::mem::size_of::<RAWINPUTHEADER>() as u32,
                );

                let mut raw_data = vec![0u8; data_size as usize];

                if GetRawInputData(
                    lparam as HRAWINPUT,
                    RID_INPUT,
                    raw_data.as_mut_ptr() as *mut c_void,
                    &mut data_size,
                    std::mem::size_of::<RAWINPUTHEADER>() as u32,
                ) == data_size
                {
                    let raw = &*(raw_data.as_ptr() as *const RAWINPUT);
                    if raw.header.dwType == RIM_TYPEMOUSE {
                        let mouse = raw.data.mouse;

                        // Raw mouse movement deltas
                        let dx = mouse.lLastX;
                        let dy = mouse.lLastY;

                        if GetAsyncKeyState(VK_LBUTTON.into()) & 0x8000u16 as i16 != 0 {
                            let sensitivity = 0.1;
                            camera.rotate(
                                dx as f32 * sensitivity,
                                -dy as f32 * sensitivity
                            );
                        }
                    }
                }
            }
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

pub fn handle_input(camera: &mut crate::helper::Camera) {
    let speed = 0.01;

    unsafe {
        // this does trigger when pressing 'W'
        if GetAsyncKeyState(VK_W.into()) & 0x8000u16 as i16 != 0 {
            camera.move_forward(speed);
        }
        if GetAsyncKeyState(VK_R.into()) & 0x8000u16 as i16 != 0 {
            camera.move_backward(speed);
        }
        if GetAsyncKeyState(VK_A.into()) & 0x8000u16 as i16 != 0 {
            camera.move_left(speed);
        }
        if GetAsyncKeyState(VK_S.into()) & 0x8000u16 as i16 != 0 {
            camera.move_right(speed);
        }

        let mut current_pos: POINT = POINT { x: 0, y: 0 };
        if GetCursorPos(&mut current_pos) != 0 {
            static mut LAST_POS: Option<(i32, i32)> = None;

            if let Some((last_x, last_y)) = LAST_POS {
                let dx = current_pos.x - last_x;
                let dy = current_pos.y - last_y;

                let sensitivity = 0.1;
                if dx != 0 || dy != 0 {
                    camera.rotate(
                        dx as f32 * sensitivity,
                        -dy as f32 * sensitivity
                    );
                }
            }
            
            std::ptr::write(&raw mut LAST_POS, Some((current_pos.x, current_pos.y)));
        }
    }
}

// Optional: Add these helper functions for better mouse control
pub fn center_cursor(hwnd: HWND) {
    unsafe {
        let mut rect = windows_sys::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        GetClientRect(hwnd, &mut rect);
        let mut point = POINT {
            x: (rect.right - rect.left) / 2,
            y: (rect.bottom - rect.top) / 2
        };
        ClientToScreen(hwnd, &mut point);
        SetCursorPos(point.x, point.y);
    }
}

pub fn toggle_cursor(show: bool) {
    unsafe {
        ShowCursor(show as i32);
    }
}