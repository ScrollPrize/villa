#include <iostream>

extern "C" __declspec(dllimport) bool exerciseScratch();

int main()
{
    if (!exerciseScratch()) {
        std::cerr << "Scratch cleanup did not complete\n";
        return 1;
    }
    std::cout << "Scratch cleanup completed; returning from main" << std::endl;
    return 0;
}
