#pragma once

namespace inplace {

__host__ __device__
void extended_gcd(int a, int b, int& gcd, int& mmi) {
    int x = 0;
    int lastx = 1;
    int y = 1;
    int lasty = 0;
    int origb = b;
    while (b != 0) {
        int quotient = a / b;
        int newb = a % b;
        a = b;
        b = newb;
        int newx = lastx - quotient * x;
        lastx = x;
        x = newx;
        int newy = lasty - quotient * y;
        lasty = y;
        y = newy;
    }
    gcd = a;
    mmi = 0;
    if (gcd == 1) {
        if (lastx < 0) {
            mmi = lastx + origb;
        } else {
            mmi = lastx;
        }
    }
}

}
