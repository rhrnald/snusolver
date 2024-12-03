#include <chrono>
#include <iostream>
#include <vector>
#include <string>
  
static std::vector<std::chrono::time_point<std::chrono::system_clock>> s;

void TIMER_PUSH() {
    s.push_back(std::chrono::system_clock::now());
}

void TIMER_START(const char* label) {
    std::cout << " -- ";
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << "    ";
    }
    std::cout << label << std::endl;
    TIMER_PUSH();
}
float TIMER_POP() {
    if (s.empty()) {
        std::cerr << "Error: TIMER_POP called without a matching TIMER_PUSH!" << std::endl;
        return -1.0f; // 에러 코드
    }
    auto end = std::chrono::system_clock::now();
    auto start = s.back();
    s.pop_back();
    return std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
}

void TIMER_END(const char* label) {
    std::cout << " -- ";
    for (size_t i = 0; i < s.size()-1; ++i) {
        std::cout << "    ";
    }
    std::cout << label << " : " << TIMER_POP() << " seconds" << std::endl;
}