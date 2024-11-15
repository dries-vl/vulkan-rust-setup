use raw_window_handle::{DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowHandle, WindowsDisplayHandle};
use std::num::NonZeroIsize;
use std::ptr::null_mut;
use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::Foundation::{HINSTANCE, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::HBRUSH;
use windows_sys::Win32::{
    System::LibraryLoader::GetModuleHandleW,
    UI::WindowsAndMessaging::*,
};

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

// Window procedure
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
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}