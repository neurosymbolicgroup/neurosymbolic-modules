package org.genesys.ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.genesys.type.AbstractType;
import org.genesys.type.IntType;
import org.genesys.type.ListType;

public class DeepCoderInputSampler implements Sampler<Object> {
	
	private final AbstractType type;
	private final DeepCoderInputSamplerParameters parameters;
	private final Random random;
	
	public DeepCoderInputSampler(AbstractType type, DeepCoderInputSamplerParameters parameters, Random random) {
		this.type = type;
		this.parameters = parameters;
		this.random = random;
	}

	@Override
	public Object sample() {
		return Collections.singletonList(this.sample(this.type));
	}
	
	private Object sample(AbstractType type) {
		if(type instanceof ListType) {
			int length = this.parameters.minLength + this.random.nextInt(this.parameters.maxLength - this.parameters.minLength);
			List<Object> list = new ArrayList<Object>();
			for(int i=0; i<=length; i++) {
				list.add(this.sample(((ListType)type).type));
			}
			return list;
		} else if(type instanceof IntType) {
			return (Integer)this.parameters.minValue + this.random.nextInt((this.parameters.maxValue - this.parameters.minValue));
		} else {
			throw new RuntimeException("Type not yet handled: " + type);
		}
	}
}
