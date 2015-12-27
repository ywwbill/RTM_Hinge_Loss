package scc;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashSet;

import lda.LDAConfig;

//Strongly Connected Component
public class SCC
{
	public int size;
	public ArrayList<ArrayList<Integer>> edges;
	public HashSet<String> visited;
	public ArrayList<ArrayList<Integer>> clusters;
	
	public int dfn[];
	public int lowLink[];
	
	public int stack[];
	public int sp;
	
	public int step;
	
	public void readGraph(String graphFileName) throws Exception
	{
		BufferedReader br=new BufferedReader(new FileReader(graphFileName));
		String line,seg[];
		while ((line=br.readLine())!=null)
		{
			seg=line.split("\t");
			int u=Integer.valueOf(seg[0]);
			int v=Integer.valueOf(seg[1]);
			edges.get(u).add(v);
			edges.get(v).add(u);
		}
		br.close();
	}
	
	public void dfs(int start)
	{	
		int tempStack[]=new int[size];
		int tempSp=0;
		tempStack[tempSp]=start;
		boolean added=true;
		
		while (tempSp>=0)
		{
			int v=tempStack[tempSp];
			
			if (added)
			{
				step++;
				dfn[v]=step;
				lowLink[v]=step;
				
				sp++;
				stack[sp]=v;
			}
			
			added=false;
			
			for (int u : edges.get(v))
			{
				if (dfn[u]==-1)
				{
					tempSp++;
					tempStack[tempSp]=u;
					added=true;
					break;
				}
				else
				{
					if (dfn[u]<dfn[v] && check(u))
					{
						lowLink[v]=Math.min(lowLink[u], lowLink[v]);
					}
				}
			}
			
			if (added) continue;
			
			if (tempSp>0)
			{
				lowLink[tempStack[tempSp-1]]=Math.min(lowLink[tempStack[tempSp-1]], lowLink[v]);
			}
			
			
			if (dfn[v]==lowLink[v])
			{
				ArrayList<Integer> temp=new ArrayList<Integer>();
				do
				{
					temp.add(stack[sp]);
					sp--;
				}while (stack[sp+1]!=v);
				clusters.add(temp);
			}
			
			tempSp--;
		}
	}
	
	public boolean check(int u)
	{
		for (int i=0; i<=sp; i++)
		{
			if (stack[i]==u)
			{
				return true;
			}
		}
		return false;
	}
	
	public void cluster()
	{
		step=-1;
		sp=-1;
		
		for (int i=0; i<size; i++)
		{
			if (dfn[i]==-1)
			{
				dfs(i);
			}
		}
	}
	
	public void writeClusters(String clusterFileName) throws Exception
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(clusterFileName));
		for (ArrayList<Integer> temp : clusters)
		{
			for (int i : temp)
			{
				bw.write(i+" ");
			}
			bw.newLine();
		}
		bw.close();
	}
	
	public void countDocs(String corpusFileName) throws Exception
	{
		BufferedReader br=new BufferedReader(new FileReader(corpusFileName));
		while (br.readLine()!=null)
		{
			size++;
		}
		br.close();
	}
	
	public void init()
	{
		dfn=new int[size];
		lowLink=new int[size];
		
		stack=new int[size];
		sp=-1;
		
		step=-1;
		
		for (int i=0; i<size; i++)
		{
			edges.add(new ArrayList<Integer>());
			dfn[i]=-1;
			lowLink[i]=-1;
		}
	}
	
	public SCC()
	{
		size=0;
		edges=new ArrayList<ArrayList<Integer>>();
		visited=new HashSet<String>();
		clusters=new ArrayList<ArrayList<Integer>>();
	}
	
	public static void main(String args[]) throws Exception
	{
		SCC scc=new SCC();
		scc.countDocs(LDAConfig.trainCorpusFileName);
		scc.init();
		scc.readGraph(LDAConfig.trainLinkFileName);
		scc.cluster();
		scc.writeClusters(LDAConfig.trainClusterFileName);
	}
}
