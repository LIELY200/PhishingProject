#include <stdio.h>

void swap(int *i, int *j);
void reset(int arr[]);
void three_place(int arr[], int length);
void reset2(int arr[]);
void swapPtr(int **iptr, int **jptr);

int main() {
	// int i = 1;
	// int j = 4;
	// int arr[] = {1, 2, 3, 5, 6, 9};
	// int *p;

	// *p = 5;


	// swap(&i, &j);

	// sizeof(int) -> 4
	// sizeof(void*) -> 8
	// sizeof(float) -> 4
	// sizeof(char) -> 1
	// sizeof(i) -> 4
	// sizeof(arr) -> 24
	// for (i = 0; i < sizeof(arr) / sizeof(arr[0]); i++) {
	// 	printf("%d\r\n", arr[i]);
	// }

	// reset2(arr);

	// for (i = 0; i < sizeof(arr) / sizeof(arr[0]); i++) {
	// 	printf("%d\r\n", arr[i]);
	// }

	// three_place(arr, sizeof(arr) / sizeof(int));
	// printf("%d\r\n", arr[3]);

	// int x = 0x000000FF;
	// char *p = (char *)&x;
	// if (p[0] == (char) 0xFF) {
	// 	printf("little endian\r\n");
	// } else {
	// 	printf("big endian\r\n`");
	// }


	int i = 4;
	int j = 8;
	int *iptr = &i;
	int *jptr = &j;
	swapPtr(&iptr, &jptr);

	printf("i ptr points to value %d and j ptr points to %d", *iptr, *jptr);

	return 0;
}

void swapPtr(int **iptrptr, int **jptrptr) {
	int *temp = *iptrptr; 
	*iptrptr = *jptrptr; 
	*jptrptr = temp;
}




// void three_place(int arr[], int length) {
// 	if(length > 3) {
// 		arr[3] = 4;
// 	}
// }


// // int* = int[] != int[N]
// void reset2(int arr[6]) {
// 	for (int i = 0; i < 6; i++) {
// 		arr[i] = 0;
// 	}
// }

// // int* = int[]
// void reset(int arr[]) {
// 	for (int i = 0 ; i < 6; i++) {
// 		arr[i] = 0;
// 	}
// }

// void swap(int *i, int *j) {
// 	int temp;

// 	temp = *i;
// 	*i = *j;
// 	*j = temp;
// }