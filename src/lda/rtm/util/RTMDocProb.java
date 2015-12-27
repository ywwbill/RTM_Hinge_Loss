package lda.rtm.util;

public class RTMDocProb implements Comparable<RTMDocProb>
{
	public int docNo;
	public double prob;
	
	public RTMDocProb(int no, double prob)
	{
		this.docNo=no;
		this.prob=prob;
	}
	
	public int compareTo(RTMDocProb o)
	{
		return -Double.compare(this.prob, o.prob);
	}
	
	public String toString()
	{
		return docNo+":"+prob;
	}
}
