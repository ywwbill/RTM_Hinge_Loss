package lda.util;

public class LDAWord implements Comparable<LDAWord>
{
	public String word;
	public int count;
	
	public LDAWord(String word, int count)
	{
		this.word=word;
		this.count=count;
	}
	
	public String toString()
	{
		return word+":"+count;
	}
	
	public int compareTo(LDAWord o)
	{
		return -Integer.compare(this.count, o.count);
	}
}
