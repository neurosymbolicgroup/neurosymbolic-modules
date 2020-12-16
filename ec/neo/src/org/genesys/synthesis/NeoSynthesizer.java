package org.genesys.synthesis;

import com.google.gson.Gson;
import com.microsoft.z3.BoolExpr;
import krangl.DataFrame;
import krangl.ReshapeKt;
import krangl.SimpleDataFrame;
import krangl.SimpleDataFrameKt;
import org.genesys.decide.Decider;
import org.genesys.decide.AbstractSolver;
import org.genesys.decide.NeoSolver;
import org.genesys.interpreter.Interpreter;
import org.genesys.language.Grammar;
import org.genesys.models.*;
import org.genesys.type.Maybe;
import org.genesys.utils.LibUtils;
import org.genesys.utils.Z3Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by ruben on 7/6/17.
 */
public class NeoSynthesizer implements Synthesizer {

    private AbstractSolver<BoolExpr, Node> solver_;

    private boolean silent_ = false;

    private boolean learning_ = true;

    private Checker checker_;

    private Interpreter interpreter_;

    private Problem problem_;

    private double totalDecide = 0.0;

    private double totalTest = 0.0;

    private HashMap<Integer, Component> components_ = new HashMap<>();

    private Gson gson = new Gson();


    public NeoSynthesizer(Grammar grammar, Problem problem, Checker checker, Interpreter interpreter, String specLoc, Decider decider) {
        solver_ = new NeoSolver(grammar, decider);
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
    }

    public NeoSynthesizer(Grammar grammar, Problem problem, Checker checker, Interpreter interpreter, int depth, String specLoc, boolean learning, Decider decider) {
        learning_ = learning;
        solver_ = new NeoSolver(grammar, depth, decider);
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
    }

    @Override
    public Node synthesize() {

        /* retrieve an AST from the solver */
        Node ast = solver_.getModel(null, false);
        int total = 0;
        int prune_concrete = 0;
        int prune_partial = 0;
        int concrete = 0;
        int partial = 0;

        while (ast != null) {
            /* do deduction */
            total++;
            if (solver_.isPartial()) partial++;
            else concrete++;

            if (!checker_.check(problem_, ast)) {
                long start = LibUtils.tick();
                if (learning_) {
                    Z3Utils z3 = Z3Utils.getInstance();
                    List<Pair<Integer, List<String>>> conflicts = z3.getConflicts();
                    ast = solver_.getCoreModel(conflicts, true, true);
                } else ast = solver_.getModel(null, true);
                long end = LibUtils.tick();
                if (solver_.isPartial()) prune_partial++;
                else prune_concrete++;
                totalDecide += LibUtils.computeTime(start, end);
                continue;
            }


            if (solver_.isPartial()) {
                //System.out.println("Partial Program: " + ast);
                ast = solver_.getModel(null, false);
                continue;
            } else {
            /* check input-output using the interpreter */
                if (verify(ast)) {
                    System.out.println("Synthesized PROGRAM: " + ast);
                    break;
                } else {
                    long start = LibUtils.tick();
                    ast = solver_.getModel(null, true);
                    long end = LibUtils.tick();
                    totalDecide += LibUtils.computeTime(start, end);
                }
            }
        }
        System.out.println("Concrete programs=: " + concrete);
        System.out.println("Partial programs=: " + partial);
        System.out.println("Decide time=:" + (totalDecide));
        System.out.println("Test time=:" + (totalTest));
        System.out.println("Total=:" + total);
        System.out.println("Prune partial=:" + prune_partial + " %=:" + prune_partial * 100.0 / partial);
        System.out.println("Prune concrete=:" + prune_concrete + " %=:" + prune_concrete * 100.0 / partial);

        return ast;
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
            try {
                Maybe<Object> tgt = interpreter_.execute(program, input);
//                System.out.println("result target:\n" + tgt.get());
//                System.out.println("expect target:\n" + output);
//                System.out.println("expect equal:\n" + ReshapeKt.hasSameContents((DataFrame) tgt.get(), (SimpleDataFrame) output));

                if (output instanceof DataFrame) {
                    boolean flag = ReshapeKt.hasSameContents((DataFrame) tgt.get(), (SimpleDataFrame) output);
                    if (!flag) {
                        passed = false;
                        break;
                    }
                } else {
                    if (!tgt.get().equals(output)) {
                        passed = false;
                        break;
                    }
                }
            } catch (Exception e) {
                if (!silent_) System.out.println("Exception= " + e);
                passed = false;
                break;
            }
        }
        long end = LibUtils.tick();
        totalTest += LibUtils.computeTime(start, end);
        return passed;
    }


}
