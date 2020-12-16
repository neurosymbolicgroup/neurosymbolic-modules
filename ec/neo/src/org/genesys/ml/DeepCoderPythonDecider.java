package org.genesys.ml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.genesys.decide.Decider;
import org.genesys.models.Example;
import org.genesys.models.Problem;
import org.genesys.models.Quad;
import org.genesys.utils.LibUtils;

public class DeepCoderPythonDecider implements Decider {
	private static final String FILENAME = "./model/tmp/deep_coder.txt";
	private static final String FUNC_FILENAME = "./model/data/deep_coder_funcs.txt";
	private static final String PYTHON_PATH_FILENAME = "./model/tmp/python_path.txt";
	private static final String COMMAND;
	
	static {
		String pythonPath = "";
		if(new File(PYTHON_PATH_FILENAME).exists()) {
			try {
				BufferedReader br = new BufferedReader(new FileReader(PYTHON_PATH_FILENAME));
				pythonPath = br.readLine();
				br.close();
			} catch(IOException e) {}
		}
		COMMAND = pythonPath + "python -m model.genesys.run";
	}
	
	private final XFeaturizer<Object> xFeaturizer;
	private final YFeaturizer yFeaturizer;
	private final List<Object> inputs = new ArrayList<Object>();
	private final List<Object> outputs = new ArrayList<Object>();
	
	private final Map<String,double[]> probabilities;
	
	public static List<String> getDeepCoderFunctions() {
		List<String> functions = new ArrayList<>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(FUNC_FILENAME));
			String line;
			while((line = br.readLine()) != null) {
				functions.add(line);
			}
			br.close();
			return functions;
		} catch(IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static DeepCoderInputSamplerParameters getDeepCoderParameters() {
		int minLength = 5;
		int maxLength = 20;
		int minValue = -256;
		int maxValue = 255;
		
		return new DeepCoderInputSamplerParameters(minLength, maxLength, maxValue, minValue);
	}

	public DeepCoderPythonDecider(Problem problem) {
		// parameters
		DeepCoderInputSamplerParameters inputSamplerParameters  = getDeepCoderParameters();
		
		// functions
		List<String> functions = getDeepCoderFunctions();
		
		// featurizers
		this.xFeaturizer = new DeepCoderXFeaturizer(inputSamplerParameters, functions);
		this.yFeaturizer  = new YFeaturizer(functions);

		//FIXME: Osbert is assuming we only have one input from ONE example.
		for(int i=0; i<5; i++) {
			Example example = problem.getExamples().get(i);
			inputs.add(LibUtils.fixGsonBug(example.getInput()));
			// Always one output table
			outputs.add(LibUtils.fixGsonBug(example.getOutput()));
		}

		this.probabilities = this.build(functions);
	}
	
	public DeepCoderPythonDecider(XFeaturizer<Object> xFeaturizer, YFeaturizer yFeaturizer, List<Object> inputs, List<Object> outputs) {
		this.xFeaturizer = xFeaturizer;
		this.yFeaturizer = yFeaturizer;
		this.inputs.addAll(inputs);
		this.outputs.addAll(outputs);
		
		this.probabilities = this.build(getDeepCoderFunctions());
	}
	
	public double getProbability(List<String> ancestors, String function) {
		if(!this.hasProbability(function)) {
			return 0.0;
		} else {
			return this.probabilities.get(Utils.toString(DeepCoderXFeaturizer.getNGram(ancestors)))[this.yFeaturizer.functionIndices.get(function)];
		}
	}
	
	public boolean hasProbability(String function) {
		return this.yFeaturizer.functionIndices.containsKey(function);
	}

	@Override
	public String decideSketch(List<String> trail, List<String> candidates, int child) { return decide(trail, candidates); }

	@Override
	public String decide(List<String> ancestors, List<String> functionChoices) {
		
		// get the most likely function
		String maxFunction = null;
		double maxProbability = -1.0;
		for(String function : functionChoices) {
			if(maxProbability <= this.getProbability(ancestors, function)) {
				maxFunction = function;
				maxProbability = this.getProbability(ancestors, function);
			}
		}
		
		if(maxFunction == null) {
			throw new RuntimeException();
		}
		
		return maxFunction;
	}
	
	private Map<String,double[]> build(List<String> functions) {
		try {
			if(DeepCoderXFeaturizer.N_GRAM_LENGTH != 2) {
				throw new RuntimeException();
			}
			
			// Step 1: Build new functions
			List<String> newFunctions = new ArrayList<String>(functions);
			newFunctions.add(DeepCoderXFeaturizer.NO_FUNCTION);
			
			// Step 2: Ensure the file exists
			File file = new File(FILENAME);
			if(!file.getParentFile().exists()) {
				file.getParentFile().mkdirs();
			}
			
			// Step 3: Build the dataset
			PrintWriter pw = new PrintWriter(new FileWriter(file));
			List<String> nGrams = new ArrayList<String>();
			for(String function0 : newFunctions) {
				for(String function1 : newFunctions) {
					// Step 3a: Build the datapoint
					List<String> nGram = Arrays.asList(new String[]{function0, function1});
					Quad<List<List<Integer>>,List<List<Integer>>,List<List<Integer>>,List<Integer>> features = this.xFeaturizer.getFeatures(this.inputs, this.outputs, nGram);
					List<List<Integer>> lists = new ArrayList<List<Integer>>();
					lists.addAll(features.t0);
					lists.addAll(features.t1);
					lists.addAll(features.t2);
					lists.add(features.t3);
					
					StringBuilder datapoint = new StringBuilder();
					datapoint.append("(");
					for(List<Integer> list : lists) {
						datapoint.append(Utils.toString(list)).append(", ");
					}
					datapoint.delete(datapoint.length()-2, datapoint.length());
					datapoint.append(")");
					
					// Step 3b: Print to test set file
					pw.println(datapoint.toString());
					
					// Step 3c: Save the n-gram
					nGrams.add(Utils.toString(nGram));
				}
			}
			pw.close();
			
			// Step 4: Execute the Python script
			Process p = Runtime.getRuntime().exec(COMMAND);
			
			// Step 5: Read the output
			BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
			String line;
			Map<String,double[]> probabilities = new HashMap<String,double[]>();
			double[] curProbabilities = null;
			int counter = 0;
			while((line = br.readLine()) != null) {
				if(line.startsWith("RESULT: ")) {
					//System.out.println("PROCESSING: " + line);
					
					// Step 5a: Build the probabilities
					String[] tokens = line.substring(9, line.length()-1).split(", ");
					curProbabilities = new double[tokens.length];
					for(int i=0; i<tokens.length; i++) {
						curProbabilities[i] = Double.parseDouble(tokens[i]);
					}
					if(curProbabilities.length != this.yFeaturizer.functions.size()) {
						throw new RuntimeException("Invalid number of probabilities!");
					}
					
					// Step 5b: Save the probabilities
					probabilities.put(nGrams.get(counter), curProbabilities);
					counter += 1;
				}
			}
			br.close();
			if(probabilities.size() != newFunctions.size() * newFunctions.size()) {
				throw new RuntimeException(probabilities.size() + ", " + newFunctions.size());
			}
			
			// Step 6: Wait for the process to finish
			p.waitFor();
			
			return probabilities;
			
		} catch(IOException | InterruptedException e) {
			throw new RuntimeException(e);
		}
	}
}
