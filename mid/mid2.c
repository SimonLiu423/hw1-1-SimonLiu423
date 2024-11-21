#include <stdio.h>

int main() {
    double s = 220 + 5 + 20 + 5 + 10 + 280 + 80 + 10 + 20 + 40 + 120 + 100 + 900 + 800 + 150 + 380;
    double p = 220 + 20 + 280 + 80 + 120 + 900 + 380;
    printf("%.2f\n", s);
    printf("%.2f\n", p);
    printf("%.2f", s / p);
    return 0;
}