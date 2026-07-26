# Deliberately standalone: links only against the acebeat library and its
# own tiny audio-decode.h (see acebeat-cli.cpp's header comment) -- zero
# acestep.cpp headers/libraries, so this stays extractable on its own.
add_executable(acebeat-cli acebeat-cli.cpp)
target_link_libraries(acebeat-cli PRIVATE acebeat)
