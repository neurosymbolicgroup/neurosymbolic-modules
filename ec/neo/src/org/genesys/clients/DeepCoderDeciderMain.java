package org.genesys.clients;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.genesys.ml.DeepCoderInputSamplerParameters;
import org.genesys.ml.DeepCoderPythonDecider;
import org.genesys.ml.DeepCoderXFeaturizer;
import org.genesys.ml.XFeaturizer;
import org.genesys.ml.YFeaturizer;

public class DeepCoderDeciderMain {
	public static void main(String[] args) {
		// parameters
		DeepCoderInputSamplerParameters inputSamplerParameters = DeepCoderPythonDecider.getDeepCoderParameters();
		
        // functions
        List<String> functions = DeepCoderPythonDecider.getDeepCoderFunctions();
        
        // featurizers
        XFeaturizer<Object> xFeaturizer = new DeepCoderXFeaturizer(inputSamplerParameters, functions);
        YFeaturizer yFeaturizer = new YFeaturizer(functions);
        
		Object input = Arrays.asList(new List[]{Arrays.asList(new Integer[]{-17, -3, 4, 11, 0, -5, -9, 13, 6, 6, -8, 11})});
		Object output = Arrays.asList(new Integer[]{-12, -20, -32, -36, -68});
		
		List<Object> inputs = Arrays.asList(new Object[]{input, input, input, input, input});
		List<Object> outputs = Arrays.asList(new Object[]{output, output, output, output, output});
        
        // decider
        DeepCoderPythonDecider decider = new DeepCoderPythonDecider(xFeaturizer, yFeaturizer, inputs, outputs);
        
        // test decider
        List<String> ancestors = new ArrayList<String>();
        for(String function : functions) {
        	double probability = decider.getProbability(ancestors, function);
        	System.out.println(function + ": " + probability);
        	ancestors.add(function);
        }
	}
}
