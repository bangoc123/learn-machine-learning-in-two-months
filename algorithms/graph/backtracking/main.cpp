//
//  main.cpp
//  Knight’s tour
//
//  Created by Ngoc Nguyen on 11/5/19.
//  Copyright © 2019 Ngoc Nguyen. All rights reserved.
//

// https://www.geeksforgeeks.org/the-knights-tour-problem-backtracking-1/

#include <iostream>
#define N 5
using namespace std;

int matrix[N][N];

bool isSafe(int x, int y) {
    return x >= 0 && y >= 0 && x <= N - 1 && y <= N-1;
}

int dfs(int x, int y, int cnt) {

    
    int xMove[8] = {  2, 1, -1, -2, -2, -1,  1,  2 };
    int yMove[8] = {  1, 2,  2,  1, -1, -2, -2, -1 };
    
    for (int i=0; i < 8; i++) {
        int nextX = x + xMove[i];
        int nextY = y + yMove[i];
        
        if (cnt == N*N) {
            return 1;
        }
        
        if (isSafe(nextX, nextY)) {
            if (matrix[nextX][nextY] == -1) {
                matrix[nextX][nextY] = cnt + 1;
                if (dfs(nextX, nextY, cnt+1) == 1) {
                    return 1;
                } else {
                    matrix[nextX][nextY] = -1; // Solution 1.
                }
            }
            
        }
        
    }
//    matrix[x][y] = -1; // Solution 2.
    return 0;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "The Knight’s tour problem\n";
        
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = -1;
        }
    }
    matrix[0][0] = 1;
    dfs(0, 0, 1);
    
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i][j] << "-";
        }
        
        cout << "\n" << endl;
    }
    
    return 0;
}
