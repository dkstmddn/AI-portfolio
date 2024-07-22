package kanye;
import java.util.*;

public class kanye {
	public static void main(String[] args) {
		int[][] a = {
				{94, 82, 100},
				{76, 75, 90},
				{92, 90, 92},
				{86, 90, 75},
				{99, 94, 100}
				};
		System.out.println("번호 국어 영어 수학  총점  평균");
		System.out.println("==============================");
		
		int sum = 0;
		double average = 0;
		
		for(int i = 0; i < a.length; i++) {
			System.out.print(i+1 + "  ");
			for(int j = 0; j < a[i].length; j++) {
				System.out.print(a[i][j] + "  ");
				sum += a[i][j];
				average = sum / 3;
			}
			System.out.print(sum + "  ");
			System.out.print(average);
			System.out.println();
			sum = 0;
			average = 0;
		}
	}
}
