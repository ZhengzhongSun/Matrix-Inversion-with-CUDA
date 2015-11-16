#include<stdio.h>
#include<math.h>
#include<iostream>
#include<fstream>
#include<time.h>

const int dimension = 1000;//dimension
float augmentedmatrix[dimension][2 * dimension];    /* 2D array declared to store augmented matrix */
#define minvalue 0.0005

int i, j, k, temp;     /* declaring counter variables for loops */

using namespace std;

//Put matrix to a txt file,j is the beginning col
void out_txt(string filename, int j_begin){
	ofstream ofile;               //define output file
	ofile.open(filename, ios::out | ios::app);     //open output file
	for (i = 0; i < dimension; i++){
		for (j = j_begin; j < 2 * dimension; j++)
			ofile << augmentedmatrix[i][j] << ",";
		ofile << "\n";
	}
	ofile.close();                //close output file
}

/*   storing augmented matrix as a matrix of dimension
(dimension)x(2*dimension) in 2D array  */
void matrix_read(){
	FILE *fp;
	int row, col;
	//char str[50];

	fp = fopen("randomMatrix_1000.txt", "r");//open matrix file
	if (fp == NULL)//open failed
		return;

	for (row = 0; row < dimension; row++){
		for (col = 0; col < 2 * dimension; col++){
			if (col < dimension){
				if (fscanf(fp, "%f,", &augmentedmatrix[row][col]) == EOF) break;//read data

				//sprintf(str, "augmentedmatrix[%d][%d]=", row, col);
				//cout << str << augmentedmatrix[row][col] << endl;
			}
			else{
				if (row == col%dimension)
					augmentedmatrix[row][col] = 1;
				else
					augmentedmatrix[row][col] = 0;

				//sprintf(str, "augmentedmatrix[%d][%d]=", row, col);
				//cout << str << augmentedmatrix[row][col] << endl;
			}
		}

		if (feof(fp)) break;//if the file is over
	}

	fclose(fp);//close file

}

/* using gauss-jordan elimination */
void gauss_jordan(){
	float temporary, r;

	for (j = 0; j<dimension; j++){
		temp = j;

		/* finding maximum jth column element in last (dimension-j) rows */

		for (i = j + 1; i<dimension; i++)
		if (augmentedmatrix[i][j]>augmentedmatrix[temp][j])
			temp = i;

		if (fabs(augmentedmatrix[temp][j])<minvalue){
			printf("\n Elements are too small to deal with !!!");
			break;
		}

		/* swapping row which has maximum jth column element */

		if (temp != j)
		for (k = 0; k<2 * dimension; k++){
			temporary = augmentedmatrix[j][k];
			augmentedmatrix[j][k] = augmentedmatrix[temp][k];
			augmentedmatrix[temp][k] = temporary;
		}

		/* performing row operations to form required identity matrix out of the input matrix */

		for (i = 0; i<dimension; i++)
		if (i != j){
			r = augmentedmatrix[i][j];
			for (k = 0; k<2 * dimension; k++)
				augmentedmatrix[i][k] -= (augmentedmatrix[j][k] / augmentedmatrix[j][j])*r;
		}
		else{
			r = augmentedmatrix[i][j];
			for (k = 0; k<2 * dimension; k++)
				augmentedmatrix[i][k] /= r;
		}

	}
}

void display(){
	/* Display augmented matrix */

	printf("\n After Gauss-Jordan elimination, augmented matrix is : \n\n");

	for (i = 0; i<dimension; i++){
		for (j = 0; j<2 * dimension; j++)
			printf("  %4.2f", augmentedmatrix[i][j]);
		printf("\n");
	}

	/* displaying inverse of the non-singular matrix */

	printf("\n\n\n The inverse of the entered non-singular matrix is : \n\n");

	for (i = 0; i<dimension; i++){
		for (j = dimension; j<2 * dimension; j++)
			printf("  %.5f", augmentedmatrix[i][j]);
		printf("\n");
	}
}

int main(){
	printf("INVERSE OF NON-SINGULAR MATRIX BY GAUSS-JORDAN ELIMINATION METHOD\n");

	/*   storing augmented matrix as a matrix of dimension
	(dimension)x(2*dimension) in 2D array  */
	matrix_read();

	//Put the augmented matrix to a txt file
	out_txt("augmentedmatrix.txt",0);

	//time start
	clock_t start, finish;
	double totaltime;
	start = clock();

	/* using gauss-jordan elimination */
	gauss_jordan();

	finish = clock();
	totaltime = (double)(finish - start);
	cout << "CPU Time - inverse:\n" << totaltime << "ms£¡" << endl;

	//Put the inverse matrix to a txt file
	out_txt("inverse.txt",dimension);

	//display matrix
	//display();

	printf("Successful!The inverse of the non-singular matrix can be saw in inverse.txt\n");

	system("pause");
	return 0;
}
