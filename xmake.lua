set_project("Solas")
set_version("0.0.0")

add_rules("mode.debug", "mode.release")

set_languages("cxx17")

target("solas")
    set_kind("shared")
    add_files("src/solas/*.cpp")
    add_includedirs("include")