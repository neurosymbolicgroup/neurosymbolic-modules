package org.genesys.clients;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.genesys.generator.Generator;
import org.genesys.ml.DeepCoderPythonDecider;
import org.genesys.models.Node;
import org.genesys.models.Trio;

public class DeepCoderDatasetGenerator {
	public static int depth = 5;
	public static int numSamples = 150000;
	public static int maxLen = 20;
	public static int numVals = 512;
	public static int numExamples = 5;
	public static int nGramLength = 2;
	
	public static List<Integer> process(Object obj) {
		List<Integer> list;
		if(obj instanceof List) {
			list = (List<Integer>)obj;
		} else if(obj instanceof Integer) {
			list = new ArrayList<Integer>();
			list.add((int)obj);
		} else {
			throw new RuntimeException();
		}
		
		List<Integer> newList = new ArrayList<Integer>();
		for(int i : list) {
			newList.add(i + numVals/2);
		}
		while(newList.size() < maxLen) {
			newList.add(numVals);
		}
		return newList;
	}
	
	public static String toStringList(List<Integer> list) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for(int i : list) {
			sb.append(i).append(", ");
		}
		sb.delete(sb.length()-2, sb.length());
		sb.append("]");
		return sb.toString();
	}
	
	public static String toStringDatapoint(List<List<Integer>> lists) {
		StringBuilder sb = new StringBuilder();
		sb.append("(");
		for(List<Integer> list : lists) {
			sb.append(toStringList(list)).append(", ");
		}
		sb.delete(sb.length()-2, sb.length());
		sb.append(")");
		return sb.toString();
	}
	
	public static Map<String,Integer> getFunctionLookup() {
		List<String> functions = DeepCoderPythonDecider.getDeepCoderFunctions();
		Map<String,Integer> lookup = new HashMap<String,Integer>();
		for(String function : functions) {
			lookup.put(function, lookup.size());
		}
		if(lookup.containsKey("")) {
			throw new RuntimeException();
		}
		lookup.put("", lookup.size());
		return lookup;
	}
	
	public static List<Integer> getFunctionIndices(List<String> functions, Map<String,Integer> lookup) {
		List<Integer> indices = new ArrayList<Integer>();
		for(String function : functions) {
			indices.add(lookup.get(function));
		}
		return indices;
	}
	
	public static List<Integer> getOneHot(int val, int len) {
		List<Integer> list = new ArrayList<Integer>();
		for(int i=0; i<len; i++) {
			list.add(0);
		}
		list.set(val, 1);
		return list;
	}
	
	private static void getNGrams(Node node, List<Integer> nGram, Map<String,Integer> lookup, List<List<Integer>> nGrams) {
		// add the current n
		LinkedList<Integer> newNGram = new LinkedList<Integer>(nGram);
		if(lookup.containsKey(node.function)) {
			newNGram.removeFirst();
			newNGram.add(lookup.get(node.function));
			nGrams.add(newNGram);
		}
		
		// recurse
		for(Node child : node.children) {
			getNGrams(child, newNGram, lookup, nGrams);
		}
	}
	
	public static List<List<Integer>> getNGrams(Node node, int n, Map<String,Integer> lookup) {
		List<Integer> nGram = new ArrayList<Integer>();
		for(int i=0; i<n; i++) {
			nGram.add(lookup.get(""));
		}
		List<List<Integer>> nGrams = new ArrayList<List<Integer>>();
		getNGrams(node, nGram, lookup, nGrams);
		return nGrams;
	}

    public static void main(String[] args) throws Exception {
    	
        String filename = "file.json";
        
        Map<String,Integer> lookup = getFunctionLookup();
        
        PrintWriter pw = new PrintWriter(new FileWriter("dataset.txt"));
        
        Generator generator = new Generator(depth, filename);
        
        for(int i=0; i<numSamples; i++) {
        	
        	if(i%1000 == 0) {
        		System.out.println(i + "/" + numSamples);
        	}
        	
	        Trio<List<List<Object>>,List<Object>,Node> program = generator.generate();
	        
	        // build input/output examples
	        
	        List<List<Integer>> firstInputs = new ArrayList<List<Integer>>();
	        List<List<Integer>> secondInputs = new ArrayList<List<Integer>>();
	        List<List<Integer>> outputs = new ArrayList<List<Integer>>();
	        
	        for(int j=0; j<numExamples; j++) {
        		firstInputs.add(process(program.t0.get(j).get(0)));
	        	if(program.t0.get(j).size() == 1) {
	        		secondInputs.add(process(new ArrayList<Integer>()));
	        	} else if(program.t0.get(j).size() == 2) {
	        		secondInputs.add(process(program.t0.get(j).get(1)));
	        	} else {
	        		throw new RuntimeException("Invalid size: " + program.t0.get(j).size());
	        	}
        		outputs.add(process(program.t1.get(j)));
	        }
	        
	        // build dsl operators
	        
	        List<List<Integer>> nGrams = getNGrams(program.t2, nGramLength+1, lookup);
	        
	        for(List<Integer> fullNGram : nGrams) {
	        	
	        	// current function
	        	List<Integer> label = getOneHot(fullNGram.get(fullNGram.size()-1), lookup.size()-1);
	        	
	        	// current n-gram
	        	LinkedList<Integer> nGram = new LinkedList<Integer>(fullNGram);
	        	nGram.removeLast();
	        	
	        	// create datapoint
	        	List<List<Integer>> datapoint = new ArrayList<List<Integer>>();
	        	for(List<Integer> list : firstInputs) {
	        		datapoint.add(list);
	        	}
	        	for(List<Integer> list : secondInputs) {
	        		datapoint.add(list);
	        	}
	        	for(List<Integer> list : outputs) {
	        		datapoint.add(list);
	        	}
	        	datapoint.add(nGram);
	        	datapoint.add(label);
	        	
	        	// add datapoint to dataset
	        	pw.println(toStringDatapoint(datapoint));
	        }
        }
        
        pw.close();

    }
}
