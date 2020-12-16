package org.genesys.ml;

public class DeepCoderInputSamplerParameters {
	public final int minLength;
	public final int maxLength;
	public final int maxValue;
	public final int minValue;
	
	public DeepCoderInputSamplerParameters(int minLength, int maxLength, int maxValue, int minValue) {
		this.minLength = minLength;
		this.maxLength = maxLength;
		this.maxValue = maxValue;
		this.minValue = minValue;
	}
}