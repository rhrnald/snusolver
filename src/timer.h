#ifndef SNUSOLVER_TIMER
#define SNUSOLVER_TIMER

#include <string>  
void TIMER_PUSH();
float TIMER_POP();

void TIMER_START(const char* label);
void TIMER_END(const char* label);

#endif