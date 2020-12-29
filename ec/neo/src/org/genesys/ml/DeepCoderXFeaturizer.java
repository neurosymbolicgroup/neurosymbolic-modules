package org.genesys.ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.genesys.models.Quad;

public class DeepCoderXFeaturizer implements XFeaturizer<Object> {
	private static final Integer NO_VALUE = null;
	public static final String NO_FUNCTION = "";
	public static final int N_GRAM_LENGTH = 2;
	
	private final Map<Integer,Integer> valueLookup = new HashMap<Integer,Integer>();
	private final Map<String,Integer> functionLookup = new HashMap<String,Integer>();
	private final DeepCoderInputSamplerParameters parameters;
	
	public DeepCoderXFeaturizer(DeepCoderInputSamplerParameters parameters, List<String> functions) {
		this.parameters = parameters;
		for(int i=this.parameters.minValue; i<=this.parameters.maxValue; i++) {
			this.valueLookup.put(i, this.valueLookup.size());
		}
		this.valueLookup.put(NO_VALUE, this.valueLookup.size());
		for(String function : functions) {
			this.functionLookup.put(function, this.functionLookup.size());
		}
		this.functionLookup.put(NO_FUNCTION, this.functionLookup.size());
	}
	
	public static List<String> getNGram(List<String> ancestors) {
		List<String> nGram = new ArrayList<String>();
		for(int i=0; i<N_GRAM_LENGTH; i++) {
			int position = ancestors.size()-i-1;
			nGram.add(position >= 0 ? ancestors.get(position) : NO_FUNCTION);
		}
		Collections.reverse(nGram);
		return nGram;
	}

	// (function n-gram, list values)
	@Override
	public Quad<List<List<Integer>>,List<List<Integer>>,List<List<Integer>>,List<Integer>> getFeatures(List<Object> inputs, List<Object> outputs, List<String> ancestors) {
		List<List<Integer>> inputValues0Features = new ArrayList<List<Integer>>();
		List<List<Integer>> inputValues1Features = new ArrayList<List<Integer>>();
		List<List<Integer>> outputValuesFeatures = new ArrayList<List<Integer>>();
		
		for(int j=0; j<5; j++) {
			// Step 1: Flatten input and output
			List<Integer> flatInput0 = new ArrayList<Integer>();
			List<Integer> flatInput1 = new ArrayList<Integer>();
			List<Integer> flatOutput = new ArrayList<Integer>();
			
			List inputList = (List)inputs.get(j);
			this.flatten(inputList.get(0), flatInput0);
			this.flatten(outputs.get(j), flatOutput);
			
			if(inputList.size() == 2) {
				this.flatten(inputList.get(1), flatInput1);
			} else if(inputList.size() == 1) {
			} else {
				throw new RuntimeException("Invalid input example size: " + inputList.size());
			}
			
			// Step 2: Featurize first part of input example
			List<Integer> inputValue0Features = new ArrayList<Integer>();
			for(int i=0; i<this.parameters.maxLength; i++) {
				Integer curValue = flatInput0.size() > i ? flatInput0.get(i) : NO_VALUE;
				inputValue0Features.add(this.valueLookup.get(curValue));
			}
			inputValues0Features.add(inputValue0Features);
			
			// Step 3: Featurize second part of input example
			List<Integer> inputValue1Features = new ArrayList<Integer>();
			for(int i=0; i<this.parameters.maxLength; i++) {
				Integer curValue = flatInput1.size() > i ? flatInput1.get(i) : NO_VALUE;
				inputValue1Features.add(this.valueLookup.get(curValue));
			}
			inputValues1Features.add(inputValue1Features);
			
			// Step 4: Featurize output example
			List<Integer> outputValueFeatures = new ArrayList<Integer>();
			for(int i=0; i<this.parameters.maxLength; i++) {
				Integer curValue = flatOutput.size() > i ? flatOutput.get(i) : NO_VALUE;
				outputValueFeatures.add(this.valueLookup.getOrDefault(curValue, this.valueLookup.get(NO_VALUE)));
			}
			outputValuesFeatures.add(outputValueFeatures);
		}
		
		// Step 5: Featurize ancestors
		List<String> nGram = getNGram(ancestors);
		List<Integer> nGramFeatures = new ArrayList<Integer>();
		for(int i=0; i<N_GRAM_LENGTH; i++) {
			nGramFeatures.add(this.functionLookup.get(nGram.get(i)));
		}
		
		return new Quad<List<List<Integer>>,List<List<Integer>>,List<List<Integer>>,List<Integer>>(inputValues0Features, inputValues1Features, outputValuesFeatures, nGramFeatures);
	}
	
	private void flatten(Object t, List<Integer> result) {
		if(t instanceof List) {
			result.addAll((List<Integer>)t);
		} else if(t instanceof Integer) {
			result.add((Integer)t);
		} else {
			throw new RuntimeException("Type not handled: " + t.getClass());
		}
	}
}
