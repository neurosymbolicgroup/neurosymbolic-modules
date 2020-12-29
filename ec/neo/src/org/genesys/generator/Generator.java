package org.genesys.generator;

import org.genesys.decide.Decider;
import org.genesys.decide.RandomDecider;
import org.genesys.interpreter.DeepCoderInterpreter;
import org.genesys.interpreter.Interpreter;
import org.genesys.language.DeepCoderGrammar;
import org.genesys.models.Example;
import org.genesys.models.Node;
import org.genesys.models.Problem;
import org.genesys.models.Trio;
import org.genesys.synthesis.Checker;
import org.genesys.synthesis.DeepCoderChecker;
import org.genesys.synthesis.GeneratorSynthesizer;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by utcs on 10/12/17.
 */
public class Generator {

    //    Array -> Array
//    Array -> Int
//    Array Int -> Array
//    Array Int -> Int
//    Array Array -> Array
//    Array Array -> Int
    public enum Options {
        A2A,A2I,AI2A,AI2I,AA2A,AA2I
    }

    private static final Random random_ = new Random();

    private Problem problem_;
    private Options option_;
    private GeneratorSynthesizer synth_;

    public Generator(int depth, String filename){
        String specLoc = "./specs/DeepCoder";

        problem_ = new Problem();
        option_ = Options.A2I;

        random();
        generateExampleTemplate();

        Problem dcProblem = getProblem();

        DeepCoderGrammar grammar = new DeepCoderGrammar(dcProblem);
        Checker checker = null;
        try {
            checker = new DeepCoderChecker(specLoc);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        Interpreter interpreter = new DeepCoderInterpreter();

        Decider decider = new RandomDecider();
        synth_ = new GeneratorSynthesizer(grammar, dcProblem, checker, interpreter, depth, specLoc, filename, decider, this);
    }

    public Trio<List<List<Object>>,List<Object>,Node> generate(){
        return synth_.synthesize();
    }

    private static <T extends Enum<?>> T randomEnum(Class<T> clazz){
        int x = random_.nextInt(clazz.getEnumConstants().length);
        return clazz.getEnumConstants()[x];
    }

    private void random(){

        Random rand = new Random();
        int  n = rand.nextInt(100) + 1;

        if (n >= 25){
            // output is an array
            Random r2 = new Random();
            int  n2 = r2.nextInt(3) + 1;
            if (n2 == 1){
                option_ = Options.A2A;
            } else if (n2 == 2){
                option_ = Options.AI2A;
            } else if (n2 == 2){
                option_ = Options.AA2A;
            }

        } else {
            // output is an int
            Random r2 = new Random();
            int  n2 = r2.nextInt(3) + 1;
            if (n2 == 1){
                option_ = Options.A2I;
            } else if (n2 == 2){
                option_ = Options.AI2I;
            } else if (n2 == 2){
                option_ = Options.AA2I;
            }
        }

        //option_ = randomEnum(Options.class);
    }

    private List<Object> generateExample(int size, int min, int max){

        List<Object> example = new ArrayList<>();

        Random random=new Random();
        for (int i = 0; i < size; i++) {
            int randomNumber = (random.nextInt(Math.abs(min+max)) + min);
            assert (randomNumber >= -min && randomNumber <= max);
            example.add(randomNumber);
        }

        return example;
    }

    private void generateExampleTemplate(){
        Example example = new Example();
        List<Object> a1 = new ArrayList<>();
        a1.add(10);
        a1.add(20);
        List<Object> a2 = new ArrayList<>();
        a2.add(10);
        a2.add(20);
        List<List<Object>> a3 = new ArrayList<>();
        List<Object> inputs = new ArrayList<>();
        List<Example> examples = new ArrayList<>();

        int output = 5;

        //System.out.println("option = " + option_);

        switch(option_){
            case A2A:
                inputs.add(a1);
                example.setInput(inputs);
                example.setOutput(a2);
                break;
            case A2I:
                inputs.add(a1);
                example.setInput(inputs);
                example.setOutput(output);
                break;
            case AI2A:
                inputs.add(a1);
                inputs.add(output);
                example.setInput(inputs);
                example.setOutput(a1);
                break;
            case AI2I:
                inputs.add(a1);
                inputs.add(output);
                example.setInput(inputs);
                example.setOutput(output);
                break;
            case AA2A:
                inputs.add(a1);
                inputs.add(a2);
                example.setInput(inputs);
                example.setOutput(a1);
                break;
            case AA2I:
                inputs.add(a1);
                inputs.add(a2);
                example.setInput(inputs);
                example.setOutput(output);
                break;
            default:
                assert(false);
        }

        examples.add(example);
        problem_.setExamples(examples);
    }

    public Problem getProblem(){
        return problem_;
    }

    public Options getOption() { return option_; }

}
