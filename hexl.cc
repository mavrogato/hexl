
#include <iostream>
#include <memory>
#include <filesystem>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <linux/input-event-codes.h>

#include <sycl/sycl.hpp>

#include <wayland-client-core.h>
#include <wayland-client-protocol.h>
#include <wayland-client.h>

#include "xdg-shell-v6-client.h"
#include "xdg-wm-base-client.h"
#include "zwp-tablet-v2-client.h"

#include "aux/algebra.hh"
#include "aux/io.hh"
#include "aux/tuple-support.hh"
#include "aux/shading.hh"

inline namespace wayland_client
{
    struct empty_type { };
    template <class> constexpr wl_interface const *const interface_ptr = nullptr;

    template <class T> concept client_like = (interface_ptr<T> != nullptr);

    template <client_like T> void (*deleter)(T*) = [](auto) { static_assert("unknown deleter"); };

    template <client_like T> struct listener_type { };
#define INTERN_CLIENT_LIKE_CONCEPT(CLIENT, DELETER, LISTENER)          \
    template <> constexpr wl_interface const *const interface_ptr<CLIENT> = &CLIENT##_interface; \
    template <> void (*deleter<CLIENT>)(CLIENT*) = DELETER;   \
    template <> struct listener_type<CLIENT> : LISTENER { };
    INTERN_CLIENT_LIKE_CONCEPT(wl_display,            wl_display_disconnect,         empty_type)
    INTERN_CLIENT_LIKE_CONCEPT(wl_registry,           wl_registry_destroy,           wl_registry_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_compositor,         wl_compositor_destroy,         empty_type)
    INTERN_CLIENT_LIKE_CONCEPT(wl_output,             wl_output_destroy,             wl_output_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_shm,                wl_shm_destroy,                wl_shm_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_seat,               wl_seat_destroy,               wl_seat_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_surface,            wl_surface_destroy,            wl_surface_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_shm_pool,           wl_shm_pool_destroy,           empty_type)
    INTERN_CLIENT_LIKE_CONCEPT(wl_buffer,             wl_buffer_destroy,             wl_buffer_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_keyboard,           wl_keyboard_destroy,           wl_keyboard_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_pointer,            wl_pointer_destroy,            wl_pointer_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_touch,              wl_touch_destroy,              wl_touch_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_callback,           wl_callback_destroy,           wl_callback_listener)
    INTERN_CLIENT_LIKE_CONCEPT(xdg_wm_base,           xdg_wm_base_destroy,           xdg_wm_base_listener)
    INTERN_CLIENT_LIKE_CONCEPT(xdg_surface,           xdg_surface_destroy,           xdg_surface_listener)
    INTERN_CLIENT_LIKE_CONCEPT(xdg_toplevel,          xdg_toplevel_destroy,          xdg_toplevel_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zxdg_shell_v6,         zxdg_shell_v6_destroy,         zxdg_shell_v6_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zxdg_surface_v6,       zxdg_surface_v6_destroy,       zxdg_surface_v6_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zxdg_toplevel_v6,      zxdg_toplevel_v6_destroy,      zxdg_toplevel_v6_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zwp_tablet_manager_v2, zwp_tablet_manager_v2_destroy, empty_type)
    INTERN_CLIENT_LIKE_CONCEPT(zwp_tablet_seat_v2,    zwp_tablet_seat_v2_destroy,    zwp_tablet_seat_v2_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zwp_tablet_tool_v2,    zwp_tablet_tool_v2_destroy,    zwp_tablet_tool_v2_listener)
#undef INTERN_CLIENT_LIKE_CONCEPT

    template <class T>
    concept client_like_with_listener = client_like<T> && !std::is_base_of_v<empty_type, listener_type<T>>;

    template <client_like T>
    [[nodiscard]] auto make_unique(T* raw = nullptr) noexcept {
        return std::unique_ptr<T, decltype (deleter<T>)>(raw, deleter<T>);
    }
    template <client_like T>
    using unique_ptr_type = decltype (make_unique<T>());

    template <class> class wrapper;

    // User defined deduction guide (Class Template Argument Deduction)
    template <class T> wrapper(T*) -> wrapper<T>;

    template <client_like T>
    class [[nodiscard]] wrapper<T> {
    public:
        wrapper(T* raw = nullptr) : ptr{make_unique(raw)}
            {
            }
        operator T*() const { return this->ptr.get(); }

    private:
        unique_ptr_type<T> ptr;
    };

    template <client_like_with_listener T>
    class [[nodiscard]] wrapper<T> {
    private:
        static constexpr auto new_default_listener() {
            static constexpr auto N = sizeof (listener_type<T>) / sizeof (void*);
            return new listener_type<T>{
                []<size_t... I>(std::index_sequence<I...>) noexcept {
                    return listener_type<T> {
                        ([]([[maybe_unused]] auto... args) noexcept {
                            (void) I;
                            // std::cerr << "Unhandled: " << typeid (listener_type<T>).name() << ':' << I
                            //           << std::tuple{args...} << std::endl;
                        })...
                    };
                }(std::make_index_sequence<N>())};
        }

    public:
        wrapper(T* raw = nullptr)
            : ptr{make_unique(raw)}
            , listener{new_default_listener()}
            {
                if (ptr != nullptr) {
                    if (0 != wl_proxy_add_listener(reinterpret_cast<wl_proxy*>(operator T*()),
                                                   reinterpret_cast<void(**)(void)>(this->listener.get()),
                                                   nullptr))
                    {
                        throw std::runtime_error("failed to add listener...");
                    }
                }
            }
        operator T*() const { return this->ptr.get(); }
        listener_type<T>* operator->() const { return this->listener.get(); }

    private:
        unique_ptr_type<T> ptr;
        std::unique_ptr<listener_type<T>> listener;
    };

    template <client_like T>
    [[nodiscard]] auto registry_bind(wl_registry* registry, uint32_t name, uint32_t version) noexcept {
        return static_cast<T*>(::wl_registry_bind(registry, name, interface_ptr<T>, version));
    }

    template <class T = aux::algebra::versor<uint8_t, 4>, wl_shm_format format = WL_SHM_FORMAT_ARGB8888>
    [[nodiscard]] auto shm_allocate_buffer(wl_shm* shm, size_t cx, size_t cy) {
        std::string_view xdg_runtime_dir = std::getenv("XDG_RUNTIME_DIR");
        if (xdg_runtime_dir.empty() || !std::filesystem::exists(xdg_runtime_dir)) {
            throw std::runtime_error("No XDG_RUNTIME_DIR settings...");
        }
        std::string tmp_path(xdg_runtime_dir);
        tmp_path += "/weston-shared-XXXXXX";
        auto fd = aux::unique_fd{::mkostemp(tmp_path.data(), O_CLOEXEC)};
        if (fd >= 0) {
            ::unlink(tmp_path.c_str());
        }
        else {
            throw std::runtime_error("mkostemp failed...");
        }
        if (::ftruncate(fd, sizeof (T)*cx*cy) < 0) {
            throw std::runtime_error("ftruncate failed...");
        }
        auto pixels = aux::unique_mmap<T>{fd, cx*cy};
        if (pixels == MAP_FAILED) {
            throw std::runtime_error("mmap failed...");
        }
        return std::tuple{
            std::move(pixels),
            wrapper(wl_shm_pool_create_buffer(wrapper(wl_shm_create_pool(shm, fd, sizeof (T) * cx*cy)),
                                              0, cx, cy, sizeof (T) * cx, format)),
        };
        // (fd closed automatically)
    }
} // ::wayland_client

inline auto lamed(auto closure) {
    static auto cache = closure;
    return [](auto... args) {
        return cache(args...);
    };
}

int main(int argc, char** argv) {
    std::filesystem::path self_path{argv[0]};

    if (argc < 2) {
        std::cerr << "Bad argument..." << std::endl;
        return 1;
    }
    std::filesystem::path input_path{argv[1]};
    if (!std::filesystem::exists(input_path)) {
        std::cerr << "Cannot open file: " << input_path << std::endl;
        return 1;
    }
    size_t input_size = std::filesystem::file_size(input_path);
    auto input_fd = aux::unique_fd{input_path.c_str(), O_RDWR};
    auto input_view = aux::unique_mmap<char>{input_fd, input_size};

    auto display = wrapper{wl_display_connect(nullptr)};
    auto registry = wrapper{wl_display_get_registry(display)};

    wrapper<wl_compositor> compositor;
    wrapper<wl_output> output;
    wrapper<wl_seat> seat;
    wrapper<wl_shm> shm;
    wrapper<xdg_wm_base> shell;

    registry->global = lamed([&](auto, auto, uint32_t name, std::string_view interface, uint32_t version) {
        if (interface == interface_ptr<wl_compositor> -> name) {
            compositor = registry_bind<wl_compositor>(registry, name, version);
        }
        else if (interface == interface_ptr<wl_output> -> name) {
            if (!output) {
                output = registry_bind<wl_output>(registry, name, version);
            }
            else {
                std::cerr << "skip secondary outputs..." << std::endl;
            }
        }
        else if (interface == interface_ptr<wl_seat> -> name) {
            seat = registry_bind<wl_seat>(registry, name, version);
        }
        else if (interface == interface_ptr<wl_shm> -> name) {
            shm = registry_bind<wl_shm>(registry, name, version);
        }
        else if (interface == interface_ptr<xdg_wm_base> -> name) {
            shell = registry_bind<xdg_wm_base>(registry, name, version);
        }
    });
    wl_display_roundtrip(display);

    size_t buffer_scale = 1;
    output->scale = lamed([&](auto, auto, auto scale) noexcept {
        std::cout << (buffer_scale = scale) << std::endl;
    });

    wrapper<wl_keyboard> keyboard;
    wrapper<wl_touch> touch;
    wrapper<wl_pointer> pointer;
    seat->capabilities = lamed([&](auto, auto, int32_t caps) noexcept {
        if (caps & WL_SEAT_CAPABILITY_KEYBOARD) keyboard = wrapper{wl_seat_get_keyboard(seat)};
        if (caps & WL_SEAT_CAPABILITY_POINTER)  pointer  = wrapper{wl_seat_get_pointer(seat)};
        if (caps & WL_SEAT_CAPABILITY_TOUCH)    touch    = wrapper{wl_seat_get_touch(seat)};
    });

    wl_display_roundtrip(display);

    auto buffer_dimension = aux::versor<size_t, 2>{1024, 768} * buffer_scale;
    auto& cx = buffer_dimension.front();
    auto& cy = buffer_dimension.back();
    auto [pixels, buffer] = shm_allocate_buffer(shm, cx, cy);
    auto que = sycl::queue();
    auto channels = [&] {
        auto deleter = [&](auto ptr) { sycl::free(ptr, que); };
        return std::unique_ptr<aux::vec4d, decltype (deleter)>{
            sycl::malloc_device<aux::vec4d>(cx*cy, que), deleter,
        };
    }();

    shell->ping = [](auto, auto shell, auto serial) noexcept {
        xdg_wm_base_pong(shell, serial);
    };
    auto surface = wrapper{wl_compositor_create_surface(compositor)};
    wl_surface_set_buffer_scale(surface, buffer_scale);
    auto xsurface = wrapper{xdg_wm_base_get_xdg_surface(shell, surface)};
    xsurface->configure = lamed([&](auto, auto, uint32_t serial) noexcept {
        xdg_surface_ack_configure(xsurface, serial);
    });
    bool running = true;
    auto toplevel = wrapper{xdg_surface_get_toplevel(xsurface)};
    xdg_toplevel_set_title(toplevel, self_path.c_str());
    xdg_toplevel_set_app_id(toplevel, self_path.c_str());
    toplevel->configure = lamed([&](auto, auto, auto w, auto h, auto) {
        buffer_dimension = aux::versor<size_t, 2>(w, h) * buffer_scale;
        std::cout << "buffer dimension configured: " << buffer_dimension << std::endl;
        if (norm(buffer_dimension)) {
            std::tie(pixels, buffer) = shm_allocate_buffer(shm, cx, cy);
            channels.reset(sycl::malloc_device<aux::vec4d>(cx*cy, que));
        }
    });
    toplevel->close = lamed([&](auto...) {
        running = false;
    });

    size_t px = 0;
    size_t py = 0;
    auto render = [&]{
        constexpr double PSI = 1.0 / std::numbers::phi;
        if (norm(buffer_dimension)) {
            //double d = std::sqrt(cx*cy/input_size);
            auto buffer_pix = sycl::buffer<aux::color, 2>{pixels.data(), {cy, cx}};
            auto buffer_chs = sycl::buffer<aux::vec4d, 2>{channels.get(), {cy, cx}};
            auto buffer_oct = sycl::buffer<char, 1>{input_view.data(), input_size};
            // que.submit([&](auto& h) noexcept {
            //     auto chs = buffer_chs.template get_access<sycl::access::mode::write>(h);
            //     h.parallel_for({cy, cx}, [=](auto idx) noexcept {
            //         chs[idx] = {0,0,0,255};
            //     });
            // });
            que.submit([&](auto& h) noexcept {
                auto chs = buffer_chs.template get_access<sycl::access::mode::read_write>(h);
                auto oct = buffer_oct.template get_access<sycl::access::mode::read>(h);
                // h.parallel_for(input_size, [=](auto idx) noexcept {
                //     double r = static_cast<double>(idx) * (cx*cy) / input_size;
                //     size_t x = std::fmod(r, cx);
                //     size_t y = (r - x) / cx;
                //     chs[{y, x}] = aux::hue((oct[idx] * 4) % 1530).template coerce<double>();
                //     // double i = idx;
                //     // double r = i / input_size;
                //     // size_t x = std::fmod(r * cy, cx);
                //     // size_t y = r * cy;
                //     // chs[{y, x}] = aux::hue((oct[i] * 4) % 1530).template coerce<double>();
                // });
                h.parallel_for({cy/2, cx/2}, [=](auto idx) noexcept {
                    double rx = std::fmod(PSI * (px + idx[1]), 1.0);
                    double ry = std::fmod(PSI * (py + idx[0]), 1.0);
                    size_t x = rx * cx;
                    size_t y = ry * cy;
                    auto& ch = chs[{y, x}];
                    // double sx = std::sqrt((double) cx / cy * input_size);
                    // double sy = std::sqrt((double) cy / cx * input_size);
                    // double xx = rx * sx;
                    // double yy = ry * sy;
                    // size_t offset = (yy * sx) + xx;
                    // ch = 0.0 * ch + 1.0 * aux::hue(4*oct[offset]).template coerce<double>();
                    ch = {192,0,0,192};
                });
                px += cx/2;
                py += cy/2;
            });
            que.submit([&](auto& h) noexcept {
                auto chs = buffer_chs.template get_access<sycl::access::mode::read>(h);
                auto pix = buffer_pix.template get_access<sycl::access::mode::write>(h);
                h.parallel_for({cy, cx}, [=](auto idx) noexcept {
                    // auto v = chs[idx];
                    // auto r_1 = aux::vec4d{
                    //     1/(v[0]+1),
                    //     1/(v[1]+1),
                    //     1/(v[2]+1),
                    //     1/(v[3]+1),
                    // };
                    // pix[idx] = aux::color{255,255,255,255} - (255.0 * r_1).template coerce<uint8_t>();
                    pix[idx] = chs[idx].template coerce<uint8_t>();
                    //pix[idx][3] = 0xff;
                });
            });
        }
    };


    aux::vec2i cursor{0, 0};
    pointer->motion = lamed([&](auto, auto, auto, auto x, auto y) {
        cursor = {wl_fixed_to_int(x), wl_fixed_to_int(y)};
    });
    pointer->button = lamed([&](auto, auto, auto serial, auto, auto button, auto state) {
        if (state == WL_POINTER_BUTTON_STATE_PRESSED) {
            switch (button) {
            case BTN_RIGHT:
                xdg_toplevel_show_window_menu(toplevel, seat, serial, cursor[0], cursor[1]);
                break;
            case BTN_MIDDLE:
                xdg_toplevel_move(toplevel, seat, serial);
                break;
            }
        }
    });
    pointer->axis = lamed([&](auto, auto, auto, auto, auto value) {
        if (value > 0) {
            ++input_size;
        }
        else {
            --input_size;
        }
        if (0 == input_size) input_size = 1;
        if (std::filesystem::file_size(input_path) < input_size) {
            input_size = std::filesystem::file_size(input_path);
        }
        std::cout << input_size << std::endl;
    });

    wrapper<wl_callback> frame{wl_surface_frame(surface)};
    frame->done = lamed([&](auto, auto, [[maybe_unused]] auto time) {
        render();
        auto rec = frame->done;
        frame = wrapper<wl_callback>{wl_surface_frame(surface)};
        frame->done = rec;
        wl_surface_damage(surface, 0, 0, cx, cy);
        wl_surface_attach(surface, buffer, 0, 0);
        wl_surface_commit(surface);
    });
    frame->done(nullptr, nullptr, 0);

    while (running) {
        if (-1 == wl_display_dispatch(display)) {
            break;
        }
    }

    return 0;
}
