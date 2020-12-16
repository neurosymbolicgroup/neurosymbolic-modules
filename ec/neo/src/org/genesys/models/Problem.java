package org.genesys.models;

import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public class Problem {
    private String name;

    private List<Example> examples;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<Example> getExamples() {
        return examples;
    }

    public void setExamples(List<Example> examples) {
        this.examples = examples;
    }

    @Override
    public String toString() {
        return "Problem{" +
                "name='" + name + '\'' +
                ", examples=" + examples +
                '}';
    }
}
