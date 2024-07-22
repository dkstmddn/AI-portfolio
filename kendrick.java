package kanye;
import java.util.*;

public class kendrick {
	public static void main(String[] args) {
		int[] arrays = new int[4]; //4칸짜리 배열 선언
		
		
		for(int i = 0; i < arrays.length; i++) {
			arrays[i] = i;
			System.out.printf("number[%d] = %d \n", i, i);
		} //0부터 3까지 반복해서 arrays의 0~3번지에 i 값 넣고 출력
		
		
		System.out.println();
		System.out.println();
		
		//배열 확인
		System.out.println(Arrays.toString(arrays));
	}
}
