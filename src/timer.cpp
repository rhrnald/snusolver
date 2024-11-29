#include <chrono>
#include <iostream>
#include <vector>
#include <string>
  
static vector<std::chrono::time_point<std::chrono::system_clock>> s;
void TIMER_PUSH() {
  s.push_back();
}
float TIMER_POP() {
  auto e = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<float>>(                   \
       (e - s.pop_back())                             \
       .count());
}
void TIMER_END(string s) {
  for(int i=0; i<s.size(); i++) printf("    ");
  std::cout << s << " : " << TIMER_POP() << std::endl;
}