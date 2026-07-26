add_executable(test-fft test-fft.cpp)
target_link_libraries(test-fft PRIVATE acebeat)
add_test(NAME test-fft COMMAND test-fft)

add_executable(test-tempo test-tempo.cpp)
target_link_libraries(test-tempo PRIVATE acebeat)
add_test(NAME test-tempo COMMAND test-tempo)

add_executable(test-key test-key.cpp)
target_link_libraries(test-key PRIVATE acebeat)
add_test(NAME test-key COMMAND test-key)
