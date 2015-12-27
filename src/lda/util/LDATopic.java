package lda.util;

public class LDATopic
{
	public int vocabCounts[];
	public int totalTokens;
	
	public LDATopic(int numVocab)
	{
		vocabCounts=new int[numVocab];
		totalTokens=0;
	}
}
