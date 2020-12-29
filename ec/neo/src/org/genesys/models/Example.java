package org.genesys.models;

import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public class Example<T> {

    private List<T> input;

    private T output;

    public List<T> getInput() {
        return input;
    }

    public T getOutput() {
        return output;
    }

    public void setInput(List<T> input) {
        this.input = input;
    }

    public void setOutput(T output) {
        this.output = output;
    }

    @Override
    public String toString() {
        return "Example{" +
                "input=" + input +
                ", output=" + output +
                '}';
    }
}
