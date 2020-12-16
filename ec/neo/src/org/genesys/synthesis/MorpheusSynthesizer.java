package org.genesys.synthesis;

import com.google.gson.Gson;
import com.microsoft.z3.BoolExpr;
import krangl.DataFrame;
import krangl.Extensions;
import krangl.ReshapeKt;
import krangl.SimpleDataFrame;
import org.genesys.decide.AbstractSolver;
import org.genesys.decide.Decider;
import org.genesys.decide.MorpheusSolver;
import org.genesys.decide.NeoSolver;
import org.genesys.interpreter.Interpreter;
import org.genesys.language.Grammar;
import org.genesys.models.*;
import org.genesys.type.Maybe;
import org.genesys.utils.LibUtils;
import org.genesys.utils.SATUtils;
import org.genesys.utils.Z3Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;

/**
 * Created by utcs on 9/11/17.
 */
public class MorpheusSynthesizer implements Synthesizer {

    private AbstractSolver<BoolExpr, Pair<Node, Node>> solver_;

    private boolean silent_ = true;

    public static boolean learning_ = true;

    private Checker checker_;

    private Interpreter interpreter_;

    private Problem problem_;

    private double totalSearch = 0.0;

    private double totalTest = 0.0;

    private double totalDeduction = 0.0;

    private HashMap<Integer, Component> components_ = new HashMap<>();

    private Gson gson = new Gson();

    public static double smt1 = 0.0;
    public static double typeinhabit = 0.0;


    public MorpheusSynthesizer(Grammar grammar, Problem problem, Checker checker, Interpreter interpreter, String specLoc, Decider decider) {
        solver_ = new MorpheusSolver(grammar, decider);
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

    public MorpheusSynthesizer(Grammar grammar, Problem problem, Checker checker, Interpreter interpreter, int depth, String specLoc, boolean learning, Decider decider) {
        learning_ = learning;
        solver_ = new MorpheusSolver(grammar, depth, decider, learning);
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
        //init equivalent class map
        Z3Utils.getInstance().initEqMap(components_.values());
    }

    @Override
    public Node synthesize() {
        long startSynth = LibUtils.tick();
        /* retrieve an AST from the solver */
        Pair<Node, Node> astPair = solver_.getModel(null, false);
        Node ast = astPair.t0;
        Node curr = astPair.t1;
        int total = 0;
        int prune_concrete = 0;
        int prune_partial = 0;
        int concrete = 0;
        int partial = 0;
        Set<String> coreCache_ = new HashSet<>();
        //Set<String> coreAst_ = new HashSet<>();

        while (ast != null) {
            /* do deduction */
            total++;
            if (solver_.isPartial()) partial++;
            else concrete++;

            //System.out.println("Checking Program: " + ast);
            long start = LibUtils.tick();
            boolean isSatisfiable = true;
            // This trick does not work well in Morpheus!
            if (solver_.isPartial())
                isSatisfiable = checker_.check(problem_, ast, curr);
            else {
                if (checker_ instanceof MorpheusChecker)
                    isSatisfiable = checker_.check(problem_, ast, curr);
            }
            long end = LibUtils.tick();
            totalDeduction += LibUtils.computeTime(start, end);

            if (!isSatisfiable) {
                if (learning_) {
                    List<List<Pair<Integer, List<String>>>> conflictsType = (List<List<Pair<Integer, List<String>>>>) checker_.learnCore();
                    Z3Utils z3 = Z3Utils.getInstance();
                    List<Pair<Integer, List<String>>> conflicts = z3.getConflicts();
                    long start2 = LibUtils.tick();
                    if (!conflictsType.isEmpty()) {
                        if (coreCache_.contains(conflictsType.toString())) {
                            astPair = solver_.getModel(null, true);
                            if (astPair == null) break;
                            ast = astPair.t0;
                            curr = astPair.t1;
                        } else {
                            astPair = solver_.getCoreModelSet(conflictsType, true, false);
                            ast = astPair.t0;
                            curr = astPair.t1;
                            coreCache_.add(conflictsType.toString());
                        }
                    } else {
                        if (conflicts.isEmpty()) {
                            astPair = solver_.getModel(null, true);
                            if (astPair == null) break;
                            ast = astPair.t0;
                            curr = astPair.t1;
                        } else {
                            astPair = solver_.getCoreModel(conflicts, true, z3.isGlobal());
                            ast = astPair.t0;
                            curr = astPair.t1;
                        }
                    }
                    long end2 = LibUtils.tick();
                    totalSearch += LibUtils.computeTime(start2, end2);
                } else {
                    long start2 = LibUtils.tick();
                    astPair = solver_.getModel(null, true);
                    ast = astPair.t0;
                    curr = astPair.t1;
                    long end2 = LibUtils.tick();
                    totalSearch += LibUtils.computeTime(start2, end2);
                }
                if (solver_.isPartial()) prune_partial++;
                else prune_concrete++;
                continue;
            }


            if (solver_.isPartial()) {
                if (!silent_) System.out.println("Partial Program: " + ast);
                long start2 = LibUtils.tick();
                solver_.cacheAST(ast.toString(), false);
                //coreAst_.add(ast.toString());
                astPair = solver_.getModel(null, false);
                ast = astPair.t0;
                curr = astPair.t1;
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
                    System.out.println("Synthesized PROGRAM: " + ast);
                    break;
                } else {
                    long start3 = LibUtils.tick();
                    astPair = solver_.getModel(null, true);
                    ast = astPair.t0;
                    curr = astPair.t1;
                    long end3 = LibUtils.tick();
                    totalSearch += LibUtils.computeTime(start3, end3);
                }
            }
        }
        long endSynth = LibUtils.tick();

        System.out.println("Concrete programs=: " + concrete);
        System.out.println("Partial programs=: " + partial);
        System.out.println("Search time=:" + (totalSearch));
        System.out.println("Deduction time=:" + (totalDeduction));
        System.out.println("Test time=:" + (totalTest));
        System.out.println("Synthesis time: " + LibUtils.computeTime(startSynth, endSynth));
        System.out.println("Total=:" + total);
        System.out.println("Prune partial=:" + prune_partial + " %=:" + prune_partial * 100.0 / partial);
        System.out.println("Prune concrete=:" + prune_concrete + " %=:" + prune_concrete * 100.0 / concrete);
        System.out.println("Learnts=:" + solver_.getLearnStats());

        System.out.println("SMT:" + smt1);
        System.out.println("Type:" + typeinhabit);

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
                if (tgt == null) {
                    passed = false;
                    break;
                }

//                System.out.println("result target:\n" + ((SimpleDataFrame)tgt.get()).getCols());
//                Extensions.print((SimpleDataFrame)tgt.get());
//                System.out.println("expected target:\n" + ((SimpleDataFrame)output).getCols());
//                Extensions.print((SimpleDataFrame)output);

                if (output instanceof DataFrame) {
                    boolean flag = ReshapeKt.hasSameContents((DataFrame) tgt.get(), (SimpleDataFrame) output);
                    if (!flag) {
                        passed = false;
                        break;
                    }
                } else {
                    if (!tgt.has() || !tgt.get().equals(output)) {
                        passed = false;
                        break;
                    }
                }
            } catch (Exception e) {
                if (!silent_) System.out.println("Exception= " + e);
                passed = false;
                //e.printStackTrace();
                //assert false;
                break;
            }
        }
        long end = LibUtils.tick();
        totalTest += LibUtils.computeTime(start, end);
        return passed;
    }

}
