package org.genesys.synthesis;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.microsoft.z3.BoolExpr;
import org.genesys.decide.AbstractSolver;
import org.genesys.decide.Decider;
import org.genesys.decide.MorpheusSolver;
import org.genesys.decide.NeoSolver;
import org.genesys.generator.SampleObject;
import org.genesys.interpreter.Interpreter;
import org.genesys.language.Grammar;
import org.genesys.models.*;
import org.genesys.models.Component;
import org.genesys.type.Maybe;
import org.genesys.utils.LibUtils;
import org.genesys.generator.Generator;

import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.List;

/**
 * Created by utcs on 9/11/17.
 */
public class GeneratorSynthesizer implements Synthesizer {

    private AbstractSolver<BoolExpr, Pair<Node,Node>> solver_;

    private boolean silent_ = true;

    private String filename_;

    private Checker checker_;

    private Interpreter interpreter_;

    private Problem problem_;

    private double totalSearch = 0.0;

    private double totalTest = 0.0;

    private double totalDeduction = 0.0;

    private HashMap<Integer, Component> components_ = new HashMap<>();

    private Gson gson = new Gson();

    private Generator generator_;

    private List<List<Object>> inputs_;
    private List<Object> outputs_;

    public GeneratorSynthesizer(Grammar grammar, Problem problem, Checker checker, Interpreter interpreter, int depth, String specLoc,
                                String filename, Decider decider, Generator generator) {
        filename_ = filename;
        solver_ = new MorpheusSolver(grammar, depth, decider, false);
        checker_ = checker;
        interpreter_ = interpreter;
        problem_ = problem;

        File[] files = new File(specLoc).listFiles();
        for (File file : files) {
            assert file.isFile() : file;
            String json = file.getAbsolutePath();
            Component comp = null;
            try {
                comp = gson.fromJson(new FileReader(json), Component.class);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            components_.put(comp.getId(), comp);
        }

        generator_ = generator;
    }

    public Object generateList(int size, int min, int max){

        List<Object> example = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            Random random=new Random();
            int randomNumber = (random.nextInt(Math.abs(min)+Math.abs(max)) + min);
            assert (randomNumber >= min && randomNumber <= max);
            example.add(randomNumber);
        }

        return example;
    }

    public Object generateInteger(int min, int max){
        Random random=new Random();
        int randomNumber = (random.nextInt(Math.abs(min)+Math.abs(max)) + min);
        return randomNumber;
    }

    public boolean isValidOutput(Object out, List<Object> current){
        boolean valid = true;
        if (out instanceof Integer){
            if ((Integer)out > 255 || (Integer)out < -256)
                valid = false;

            for (Object i : current){
                if ((Integer)out == (Integer)i) {
                    valid = false;
                    break;
                }
            }

        } else if (out instanceof List){
            if (((List)out).isEmpty() || ((List)out).size() == 20)
                valid = false;

            if (valid) {
                Set<Integer> contents = new HashSet<>();
                for (Object o : (List) out) {
                    if ((Integer) o > 255 || (Integer) o < -256) {
                        valid = false;
                        break;
                    } else {
                        contents.add((Integer)o);
                    }
                }
                if (contents.size() <= 1)
                    valid = false;
            }
        }

        return valid;
    }

    public List<String> getComponents(Node ast){

        List<String> components = new ArrayList<>();
        Deque<Node> working = new LinkedList<>();
        working.add(ast);

        while (!working.isEmpty()){
            Node node = working.pollFirst();
            if (!node.function.contains("input") && !node.function.contains("root"))
                components.add(node.function);
            for (Node n : node.children){
                working.add(n);
            }
        }

        Collections.reverse(components);
        return components;
    }

    @Override
    public Trio<List<List<Object>>,List<Object>,Node> synthesize() {

        /* retrieve an AST from the solver */
        Node ast = solver_.getModel(null, false).t0;
        int total = 0;
        int prune_concrete = 0;
        int prune_partial = 0;
        int concrete = 0;
        int partial = 0;
        Set<String> coreCache_ = new HashSet<>();

        while (ast != null) {
            /* do deduction */
            total++;
            if (solver_.isPartial()) partial++;
            else concrete++;

            long start = LibUtils.tick();
            boolean isSatisfiable = true;
            long end = LibUtils.tick();
            totalDeduction += LibUtils.computeTime(start, end);

            if (solver_.isPartial()) {
                if (!silent_) System.out.println("Partial Program: " + ast);
                long start2 = LibUtils.tick();
                solver_.cacheAST(ast.toString(), false);
                ast = solver_.getModel(null, false).t0;
                long end2 = LibUtils.tick();
                totalSearch += LibUtils.computeTime(start2, end2);
                continue;
            } else {
                /* check input-output using the interpreter */
                long start2 = LibUtils.tick();
                boolean isCorrect = verify(ast);
                long end2 = LibUtils.tick();
                totalTest += LibUtils.computeTime(start2, end2);

                if (isCorrect) {
                    //System.out.println("Synthesized PROGRAM: " + ast);
                    break;
                } else {
                    long start3 = LibUtils.tick();
                    ast = solver_.getModel(null, true).t0;
                    long end3 = LibUtils.tick();
                    totalSearch += LibUtils.computeTime(start3, end3);
                }
            }
        }

        Trio res = new Trio(inputs_,outputs_,ast);
        return res;
    }

    public void writeToJSON(String filename, Node program){

        SampleObject sample = new SampleObject(filename,inputs_,outputs_, program);
        Gson gson = new GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create();
        String json = gson.toJson(sample);
//        System.out.println(json);

        try (FileWriter writer = new FileWriter(filename)) {

            gson.toJson(sample, writer);

        } catch (IOException e) {
            e.printStackTrace();
        }


    }


    /* Verify the program using I-O examples. */
    private boolean verify(Node program) {
        long start = LibUtils.tick();
        boolean passed = true;
        if (!silent_) System.out.println("Program: " + program);
        for (Example example : problem_.getExamples()) {
            //FIXME:lets assume we only have at most two input tables for now.
            Object input = LibUtils.fixGsonBug(example.getInput());
            // Always one output table
            Object output = LibUtils.fixGsonBug(example.getOutput());

            int itn = 0;
            List<List<Object>> inputs = new ArrayList<>();
            List<Object> outputs = new ArrayList<>();
            int min = -128;
            int max = 128;

            while (inputs.size() != 5) {
               try {

                   Generator.Options opt = generator_.getOption();
                   List<Object> in = new ArrayList<>();

                   switch(opt){
                       case A2A:
                       case A2I:
                           in.add(generateList(20, min, max));
                           break;
                       case AI2A:
                       case AI2I:
                           in.add(generateList(20, min, max));
                           in.add(generateInteger(min,max));
                           break;
                       case AA2A:
                       case AA2I:
                           in.add(generateList(20, min, max));
                           in.add(generateList(20, min, max));
                           break;
                       default:
                           assert(false);
                   }


                    Maybe<Object> tgt = interpreter_.execute(program, in);
                    if (isValidOutput(tgt.get(),outputs)){
                        inputs.add(in);
                        outputs.add(tgt.get());
                        itn = 0;
                    }

                } catch (Exception e) {
                    if (!silent_) System.out.println("Exception= " + e);
                    //passed = false;
                    //e.printStackTrace();
                    //assert false;
                    //break;
                }
                itn++;
                if (itn == 100){
                   passed = false;
                   break;
                }
            }

            if (inputs.size() < 5)
                passed = false;

            if (passed){
                inputs_ = inputs;
                outputs_ = outputs;
//                System.out.println("inputs = " + inputs);
//                System.out.println("outputs = " + outputs);
                writeToJSON(filename_, program);
            }
        }
        long end = LibUtils.tick();
        totalTest += LibUtils.computeTime(start, end);
        return passed;
    }

}

