package org.genesys.generator;

import java.util.List;
import java.util.ArrayList;

/**
 * Created by utcs on 10/14/17.
 */
public class Example {

    List<Object> input = new ArrayList<>();
    Object output;

    public Example (List<Object> inputs, Object output){
        this.input = inputs;
        this.output = output;
    }

}
