package org.genesys.decide;

import com.google.gson.Gson;
import org.genesys.language.Grammar;
import org.genesys.language.ToyGrammar;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.sat4j.minisat.constraints.MixedDataStructureDanielWL;
import org.sat4j.minisat.core.Constr;
import org.sat4j.minisat.core.DataStructureFactory;
import org.sat4j.minisat.core.Solver;
import org.sat4j.minisat.learning.MiniSATLearning;
import org.sat4j.minisat.orders.RSATPhaseSelectionStrategy;
import org.sat4j.minisat.orders.VarOrderHeap;
import org.sat4j.minisat.restarts.Glucose21Restarts;
import org.sat4j.minisat.restarts.MiniSATRestarts;
import org.sat4j.core.VecInt;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.Lbool;
import org.sat4j.specs.TimeoutException;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;


/**
 * Created by yufeng on 5/31/17.
 */
public class SATSolver<C, T> implements AbstractSolver<C, T> {

    private Decider decider_;
    private Solver solver_;
    private boolean allVariablesAssigned_;
    private Map<String,Integer> production2variable_;

    public SATSolver(Grammar g, Decider d) {
        decider_ = d;
        allVariablesAssigned_ = false;
        production2variable_ = new HashMap<String,Integer>();
        solver_ = defaultSolver();
    }

    @Override
    public boolean isPartial() { return false; }

    @Override
    public void cacheAST(String program, boolean block) {};

    @Override
    public T getCoreModel(List<Pair<Integer, List<String>>> core, boolean block, boolean global) {
        return null;
    }

    @Override
    public T getCoreModelSet(List<List<Pair<Integer, List<String>>>> core, boolean block, boolean global) {
        return null;
    }

    @Override
    public ArrayList<Double> getLearnStats() {return null; };

    @Override
    public T getModel(C core, boolean block) {
        return null;
    }

    public static Solver defaultSolver(){
        // Using the default infrastructure from SAT4J
        MiniSATLearning<DataStructureFactory> learning = new MiniSATLearning<DataStructureFactory>();
        Solver<DataStructureFactory> solver = new Solver<DataStructureFactory>(
                learning, new MixedDataStructureDanielWL(), new VarOrderHeap(), new MiniSATRestarts());
        solver.setSimplifier(solver.EXPENSIVE_SIMPLIFICATION);
        solver.setOrder(new VarOrderHeap(new RSATPhaseSelectionStrategy()));
        solver.setRestartStrategy(new Glucose21Restarts());
        solver.setLearnedConstraintsDeletionStrategy(solver.glucose);
        solver.setTimeout(36000);
        return solver;
    }

    // FIXME: we need to track if we have any decision left
    public boolean allVariablesAssigned(){
        return allVariablesAssigned_;
    }

    // FIXME: save the model into a private datastructure
    public void saveModel(int[] model){

    }

    public int posLit(int var){
        return var << 1;
    }

    public int negLit(int var) {
        return var << 1 ^ 1;
    }

    public void createVariables(){
        // FIXME: build the AST and create the variables for each node
    }

    public void createConstraints(){
        // FIXME: create constraints for the grammar
    }

    /**
     * Search procedure for the SAT solver to be adapted by the genesys loop.
     * Based on the search code from the SAT4J SAT solver.
     */
    public Lbool search(Solver solver){

        Lbool res = Lbool.UNDEFINED;
        int rootLevel = solver.decisionLevel();
        int backjumpLevel = 0;
        org.sat4j.minisat.core.Pair analysisResult = new org.sat4j.minisat.core.Pair();

        Map<String,Integer> prod2var = new HashMap<String,Integer>();
        List<String> prodTrail = new ArrayList<String>();
        List<String> candidates = new ArrayList<String>();

        do {
            solver.slistener.beginLoop();
            Constr conflict = solver.propagate();
            if (conflict == null){
                // no conflict found
                if (allVariablesAssigned()){
                    // if all variables are assigned save model
                    solver.modelFound();
                    saveModel(solver.model());
                    // FIXME: populate AST with the model?
                    res = Lbool.TRUE;
                } else {
                    // new decision
                    String decString = decider_.decide(prodTrail,candidates);
                    assert (production2variable_.containsKey(decString));
                    // pick the positive polarity
                    int dec = posLit(production2variable_.get(decString));
                    boolean ret = solver.assume(dec);
                    assert ret;
                }
            } else {
                // conflict found
                solver.slistener.conflictFound(conflict, solver.decisionLevel(),
                        solver.trail.size());
                if (solver.decisionLevel() == rootLevel){
                    // unsat
                    res = Lbool.FALSE;
                } else {
                    int conflictTrailLevel = solver.trail.size();
                    // analyze conflict
                    try {
                        solver.analyze(conflict, analysisResult);
                    } catch (TimeoutException e) {
                        // we should never have a timeout
                    }
                    assert analysisResult.backtrackLevel < solver.decisionLevel();
                    backjumpLevel = Math.max(analysisResult.backtrackLevel, rootLevel);
                    solver.cancelUntil(backjumpLevel);
                    assert solver.decisionLevel() >= rootLevel
                            && solver.decisionLevel() >= analysisResult.backtrackLevel;

                    if (analysisResult.reason == null) {
                        res = Lbool.FALSE;
                    }
                    solver.record(analysisResult.reason);
                    solver.restarter.newLearnedClause(analysisResult.reason,
                            conflictTrailLevel);
                    analysisResult.reason = null;
                }

            }

        } while (res == Lbool.UNDEFINED);

        return res;
    }

    public Solver getSolver(){
        return solver_;
    }

    public static void main(String[] args) throws FileNotFoundException {

        Gson gson = new Gson();
        ToyGrammar toyGrammar = gson.fromJson(new FileReader("./grammar/Toy.json"), ToyGrammar.class);
        toyGrammar.init();

        RandomDecider decider = new RandomDecider();
        SATSolver satSolver = new SATSolver(toyGrammar, decider);


        satSolver.getSolver().newVar(2);
        try {
            satSolver.getSolver().addClause(new VecInt(new int[] {-1,2}));
            satSolver.getSolver().addClause(new VecInt(new int[] {2,1}));
            satSolver.getSolver().addClause(new VecInt(new int[] {-1}));
            satSolver.getSolver().addClause(new VecInt(new int[] {2}));
        } catch (ContradictionException e) {
            e.printStackTrace();
        }
        //boolean res = false;
        satSolver.getSolver().propagate();
//        try {
            //res = satSolver.getSolver().isSatisfiable();
//        } catch (TimeoutException e) {
//            e.printStackTrace();
//        }
//        System.out.println("res = " + res);
    }
}
