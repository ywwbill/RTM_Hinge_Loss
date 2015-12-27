package utility;

import java.io.BufferedReader;
import java.io.BufferedWriter;

public class Util
{
	public static void print(Object obj)
	{
		System.out.print(obj);
	}
	
	public static void println(Object obj)
	{
		System.out.println(obj);
	}
	
	public static void println()
	{
		System.out.println();
	}
	
	public void printMatrix(int matrix[][])
	{
		for (int i=0; i<matrix.length; i++)
		{
			for (int j=0; j<matrix[i].length; j++)
			{
				Util.print(matrix[i][j]+" ");
			}
			Util.println();
		}
	}
	
	public static void readMatrix(BufferedReader br, double matrix[][]) throws Exception
	{
		readMatrix(br, matrix, matrix.length, matrix[0].length);
	}
	
	public static void readMatrix(BufferedReader br, double matrix[][], int dim1, int dim2) throws Exception
	{
		String line,seg[];
		for (int i=0; i<dim1; i++)
		{
			line=br.readLine();
			seg=line.split(" ");
			for (int j=0; j<dim2; j++)
			{
				matrix[i][j]=Double.valueOf(seg[j]);
			}
		}
	}
	
	public static void readVector(BufferedReader br, double vector[]) throws Exception
	{
		readVector(br, vector, vector.length);
	}
	
	public static void readVector(BufferedReader br, double vector[], int dim) throws Exception
	{
		String line;
		for (int i=0; i<dim; i++)
		{
			line=br.readLine();
			vector[i]=Double.valueOf(line);
		}
	}
	
	public static void writeMatrix(BufferedWriter bw, int matrix[][]) throws Exception
	{
		writeMatrix(bw, matrix, matrix.length, matrix[0].length);
	}
	
	public static void writeMatrix(BufferedWriter bw, int matrix[][], int dim1, int dim2) throws Exception
	{
		for (int i=0; i<dim1; i++)
		{
			for (int j=0; j<dim2; j++)
			{
				bw.write(matrix[i][j]+" ");
			}
			bw.newLine();
		}
	}
	
	public static void writeMatrix(BufferedWriter bw, double matrix[][]) throws Exception
	{
		writeMatrix(bw, matrix, matrix.length, matrix[0].length);
	}
	
	public static void writeMatrix(BufferedWriter bw, double matrix[][], int dim1, int dim2) throws Exception
	{
		for (int i=0; i<dim1; i++)
		{
			for (int j=0; j<dim2; j++)
			{
				bw.write(matrix[i][j]+" ");
			}
			bw.newLine();
		}
	}
	
	public static void writeVector(BufferedWriter bw, double vector[]) throws Exception
	{
		writeVector(bw, vector, vector.length);
	}
	
	public static void writeVector(BufferedWriter bw, int vector[]) throws Exception
	{
		writeVector(bw, vector, vector.length);
	}
	
	public static void writeVector(BufferedWriter bw, double vector[], int dim) throws Exception
	{
		for (int i=0; i<dim; i++)
		{
			bw.write(vector[i]+"");
			bw.newLine();
		}
	}
	
	public static void writeVector(BufferedWriter bw, int vector[], int dim) throws Exception
	{
		for (int i=0; i<dim; i++)
		{
			bw.write(vector[i]+"");
			bw.newLine();
		}
	}
}
