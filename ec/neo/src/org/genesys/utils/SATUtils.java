package org.genesys.utils;

import org.genesys.ml.Utils;
import org.sat4j.core.Vec;
import org.sat4j.core.VecInt;
import org.sat4j.minisat.constraints.MixedDataStructureDanielWL;
import org.sat4j.minisat.core.DataStructureFactory;
import org.sat4j.minisat.core.Solver;
import org.sat4j.minisat.core.Constr;
import org.sat4j.minisat.learning.MiniSATLearning;
import org.sat4j.minisat.orders.RSATPhaseSelectionStrategy;
import org.sat4j.minisat.orders.VarOrderHeap;
import org.sat4j.minisat.restarts.Glucose21Restarts;
import org.sat4j.minisat.restarts.MiniSATRestarts;
import org.sat4j.specs.*;

import java.util.Deque;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.LinkedList;

import org.genesys.models.Pair;

/**
 * Created by utcs on 7/7/17.
 */
public class SATUtils {

    /* making propagate faster */
    public enum ClauseType {TYPEIH, ASSIGNMENT, SKTASSIGNMENT, EQCLASS, GLOBAL, LEARNT, LOCAL};
    private Set<IConstr> assignments_ = new HashSet<>();
    private HashMap<IConstr, VecInt> assignment2clause_ = new HashMap<>();
    private Vec<IConstr> eqlearnts_ = new Vec<>();
    private List<Pair<VecInt, List<Pair<Integer,String>>>> allEqLearnts_ = new ArrayList<>();

    //private Set<IConstr> local_ = new HashSet<>();
    private IVec<IConstr> local_ = new Vec<>();

    private static SATUtils instance = null;

    private List<Boolean> variablesOccurs = new ArrayList<>();

    private Solver solver_ = null;

    private int nbVars = 0;

    private boolean init = false;

    private List<Pair<IConstr,Integer>> learnts_ = new ArrayList<>();

    private Deque<Integer> freeVariables_ = new LinkedList<>();
    private ArrayList<Integer> usedVariables_ = new ArrayList<>();

    private int globalClauses_ = 0;

    public void createSolver() {
        MiniSATLearning<DataStructureFactory> learning = new MiniSATLearning<DataStructureFactory>();
        Solver<DataStructureFactory> solver = new Solver<DataStructureFactory>(
                learning, new MixedDataStructureDanielWL(), new VarOrderHeap(), new MiniSATRestarts());
        solver.setSimplifier(solver.SIMPLE_SIMPLIFICATION);
        solver.setOrder(new VarOrderHeap(new RSATPhaseSelectionStrategy()));
        solver.setRestartStrategy(new Glucose21Restarts());
        solver.setLearnedConstraintsDeletionStrategy(solver.glucose);
        solver.setTimeout(36000);
        solver_ = solver;
    }

    public void createVars(int vars) {
//        for (int i = 0; i < vars; i++)
//            variablesOccurs.add(false);
        //solver_.newVar(vars);
        // FIXME: just creating a lot of variables in advance
        solver_.newVar(vars);
        //solver_.newVar(1000000);
        nbVars = vars;

//        nbVars = vars;
//
//        // SAT4J requires that a variable exists in the formula; otherwise it may lead to problems
//        // hack to circunvent this problem -- uses a dummy variable
//        for (int i = 1; i <= vars; i++) {
//            boolean res = addClause(new VecInt(new int[]{i, vars + 1}));
//        }
//
//        try {
//            // hack to initialize SAT4J data structures
//            solver_.isSatisfiable();
//        } catch (TimeoutException e) {
//            e.printStackTrace();
//        }
    }

    public void updateEqLearnts(List<Pair<VecInt, List<Pair<Integer,String>>>> eq){
        allEqLearnts_.addAll(eq);
    }

    public boolean addEqLearnts(List<Pair<Integer,String>> skt, List<Integer> sktid){
        boolean conflict = false;

        for (Pair<VecInt, List<Pair<Integer,String>>> p : allEqLearnts_){
            boolean contains = true;
            for (Pair<Integer,String> pair : p.t1){
                if (sktid.contains(pair.t0) && !skt.contains(pair)){
                    contains = false;
                    break;
                }
            }
            if (contains) {
                // learnt clause is relevant to this sketch
                conflict &= addClause(p.t0, ClauseType.EQCLASS);
            }
        }
        return conflict;
    }

    public void cleanEqLearnts(){
        for (int i = 0; i < eqlearnts_.size(); i++){
            solver_.removeConstr(eqlearnts_.get(i));
        }
        eqlearnts_.clear();
    }

    public Constr propagate(){

        if (!init) {
            try {
                // hack to initialize SAT4J data structures
                boolean res = solver_.initNeo();
                assert (res);
            } catch (TimeoutException e) {
                e.printStackTrace();
            }
            init = true;
        }

        return solver_.propagate();
    }

    public int getNbVars() {
        return nbVars;
    }

    public boolean varOccurs(int var) {
        assert (var < variablesOccurs.size());
        return variablesOccurs.get(var);
    }

    public Solver getSolver() {
        return solver_;
    }


    public boolean blockTrail(VecInt trail) {
        assert (solver_ != null);
        assert (!trail.isEmpty());

//        System.out.println("block trail= " + trail);
//        System.out.println("lvl = " + solver_.decisionLevel());

//        VecInt clause = new VecInt();
//        for (int i = 0; i < trail.size(); i++) {
//            clause.push(-trail.get(i));
//        }

        boolean conflict = false;
        conflict = addClause(trail, ClauseType.ASSIGNMENT);
//        if (solver_.decisionLevel() == 0) conflict = addClause(clause);
//        else conflict = addClauseOnTheFly(clause);

        return conflict;
    }

    public void cleanVariables(){
        for (Integer i : usedVariables_){
            if (solver_.truthValue(i) == Lbool.UNDEFINED)
                freeVariables_.push(i);
        }
        usedVariables_.clear();
    }


    public boolean learnCoreGlobal(List<List<Integer>> core){
        boolean conflict = false;

        // create k auxiliary variables
        List<Integer> aux = new ArrayList<>();

//        for (int i = nbVars+1; i <= nbVars+core.size(); i++)
//            aux.add(i);

        for (int i = 1; i <= core.size(); i++){
            if (!freeVariables_.isEmpty()){
                int v = freeVariables_.poll();
                aux.add(v);
                usedVariables_.add(v);
            } else {
                aux.add(++nbVars);
                usedVariables_.add(nbVars);
            }
        }

        //nbVars = nbVars+core.size();
        assert (aux.size() == core.size());

        assert (nbVars < 1000000);

        // FIXME: problem with increasing the number of variables in SAT4J
//        solver_.newVar(solver_.nVars() + core.size());
//        System.out.println("nbVars= " + solver_.nVars());

        // equivalence between auxiliary variables and core variables
        int pos = 0;
        for (List<Integer> p : core){
            VecInt eqclause = new VecInt();
            for (Integer l : p){
                //conflict = conflict || addClause(new VecInt(new int[]{-aux.get(pos),l}));
                conflict = conflict || addClause(new VecInt(new int[]{aux.get(pos),-l}));
                eqclause.push(l);
            }
            eqclause.push(-aux.get(pos));
            conflict = conflict || addClause(eqclause);
            pos++;
        }

        VecInt clause = new VecInt();
        for (Integer l : aux){
            clause.push(-l);
        }
        conflict = conflict || addClause(clause);
        assert (!conflict);


        return conflict;
    }

    public boolean learnCoreLocal(List<List<Integer>> core, int line){
        boolean conflict = false;

        // create k auxiliary variables
        List<Integer> aux = new ArrayList<>();
        //for (int i = solver_.nVars()+1; i <= solver_.nVars()+core.size(); i++)

        for (int i = 1; i <= core.size(); i++){
            if (!freeVariables_.isEmpty()){
                int v = freeVariables_.poll();
                aux.add(v);
                usedVariables_.add(v);
            } else {
                aux.add(++nbVars);
                usedVariables_.add(nbVars);
            }
        }

        //nbVars = nbVars+core.size();
        assert (aux.size() == core.size());

        assert (nbVars < 1000000);


        // FIXME: problem with increasing the number of variables in SAT4J
//        solver_.newVar(solver_.nVars() + core.size());
//        System.out.println("nbVars= " + solver_.nVars());

        // equivalence between auxiliary variables and core variables
        int pos = 0;
        for (List<Integer> p : core){
            VecInt eqclause = new VecInt();
            for (Integer l : p){
                //conflict = conflict || addClause(new VecInt(new int[]{-aux.get(pos),l}));
                conflict = conflict || addLearnt(new VecInt(new int[]{aux.get(pos),-l}),line);
                eqclause.push(l);
            }
            eqclause.push(-aux.get(pos));
            conflict = conflict || addLearnt(eqclause,line);
            pos++;
        }

        VecInt clause = new VecInt();
        for (Integer l : aux){
            clause.push(-l);
        }
        conflict = conflict || addLearnt(clause,line);
        assert (!conflict);

        return conflict;
    }

    public boolean allVariablesAssigned() {
        assert (solver_ != null);

        boolean ok = true;
        for (int i = 0; i <= solver_.nVars(); i++) {
            if (solver_.truthValue(posLit(i)) == Lbool.UNDEFINED) {
                ok = false;
                break;
            }
        }
        return ok;
    }

    public boolean addAMK(VecInt clause, int k){
        assert (solver_ != null);

        boolean conflict = false;
        try {
            solver_.addAtMost(clause, k);
        } catch (ContradictionException e) {
            conflict = true;
        }
        return conflict;
    }

    public boolean addEO(VecInt clause, int k){
        assert (solver_ != null);

        boolean conflict = false;
        try {
            solver_.addExactly(clause, k);
        } catch (ContradictionException e) {
            conflict = true;
        }
        return conflict;
    }

    public boolean addClause(VecInt clause, ClauseType ct) {
        assert (solver_ != null);

        boolean conflict = false;
        try {
            IConstr c = solver_.addClause(clause);
            // c can be null if it is already satisfied, e.g. we learned an unit clause (rare but possible)
            if (c != null) {
                if (ct.equals(ClauseType.GLOBAL)){
                    globalClauses_++;
//                    System.out.println("GLOBAL = " + globalClauses_);
                }

                if (ct.equals(ClauseType.LOCAL)){
                    local_.push(c);
                }

                if (ct.equals(ClauseType.ASSIGNMENT)) {
                    assignment2clause_.put(c,clause);
                    assignments_.add(c);
                }

                if (ct.equals(ClauseType.EQCLASS)) {
                    eqlearnts_.push(c);
                }

                if (ct.equals(ClauseType.SKTASSIGNMENT)) {

                    List<IConstr> toberemoved = new ArrayList<>();
                    for (IConstr ctr : assignments_) {
                        VecInt cls = assignment2clause_.get(ctr);
                        if (cls.size() < clause.size()){
                            // keep the clause until we change the first component?
                            if (cls.get(0) != clause.get(0)){
                                toberemoved.add(ctr);
                            }
                        } else {
                            toberemoved.add(ctr);
                        }
                    }

                    for (IConstr ctr : toberemoved) {
                        solver_.removeConstr(ctr);
                        assignments_.remove(ctr);
                        assignment2clause_.remove(ctr);
                    }
                    //assignments_.clear();
                }
            }
        } catch (ContradictionException e) {
            conflict = true;
        }
        return conflict;
    }

    public boolean addClause(VecInt clause) {
        assert (solver_ != null);

        boolean conflict = false;
        try {
            solver_.addClause(clause);
        } catch (ContradictionException e) {
            conflict = true;
        }
        return conflict;
    }


    public boolean addLearnt(VecInt clause, int line){
        assert (solver_ != null);

        boolean conflict = false;
        try {

            IConstr c = solver_.addClause(clause);
            // can the clause be satisfied?
            if (c != null)
                learnts_.add(new Pair<IConstr,Integer>(c,line));
        } catch (ContradictionException e) {
            conflict = true;
        }
        return conflict;
    }

    public void cleanLearnts(){
        for (Pair<IConstr,Integer> c : learnts_){
            solver_.removeConstr(c.t0);
        }
        learnts_.clear();
    }

    public void cleanLocals(){

        for (int i = 0; i < local_.size(); i++){
            solver_.removeConstr(local_.get(i));
        }
        local_.clear();

//        Vec<IConstr> half = new Vec<>();
//
//        for (int i  = 0; i < local_.size()/2; i++){
//            solver_.removeConstr(local_.get(i));
//        }
//
//        for (int i = local_.size()/2; i < local_.size(); i++){
//            half.push(local_.get(i));
//        }
//        local_.clear();
//        half.copyTo(local_);
    }

    public void removeClause(IConstr c){
        solver_.removeConstr(c);
    }

    public void cleanLearnts(int line){
        List<Pair<IConstr,Integer>> tmp = new ArrayList<>();
        for (Pair<IConstr,Integer> c : learnts_){
            if (c.t1 > line) {
                tmp.add(c);
                solver_.removeConstr(c.t0);
            }
        }
        learnts_.removeAll(tmp);
        //System.out.println("#learnts = " + learnts_.size());
    }

    public boolean addClauseOnTheFly(VecInt clause) {
        assert (solver_ != null);

            boolean conflict = false;

            int [] c = new int[clause.size()];
            int asserting = 0;
            int nb_asserting = 0;
            boolean satisfied = false;
            for (int i = 0; i < clause.size(); i++) {
                c[i] = clause.get(i);
                int var = c[i];
                if (asserting == 0 && solver_.truthValue(var) == Lbool.UNDEFINED){
                    nb_asserting++;
                    asserting = var;
                }
            }

            solver_.addClauseOnTheFly(c);
            solver_.qhead = solver_.trail.size(); // Why doesn't SAT4J does this?
//            System.out.println("current ctr= " + solver_.getIthConstr(solver_.nConstraints()-1));
//            System.out.println("assuming = " + asserting);
            Constr ctr = solver_.propagate();
            if (ctr != null || nb_asserting == 0) {
                conflict = true;
                assert(false);
            } else {
                assert (nb_asserting == 1);
                if (asserting < 0) solver_.assume(negLit(-asserting));
                else solver_.assume((posLit(asserting)));
            }

            return conflict;
    }

    public int posLit(int var) {
        return var << 1;
    }

    public int negLit(int var) {
        return var << 1 ^ 1;
    }

    public static SATUtils getInstance() {
        if (instance == null) {
            instance = new SATUtils();
        }
        return instance;
    }

    public void printTrail(){
        System.out.println("SAT4J trail= " + solver_.trail);
    }

    public boolean addAMO(VecInt lits) {

        assert (false);
        assert (lits.size() != 0);
        boolean conflict = false;

        if (lits.size() == 1) {
            if (addClause(lits))
                conflict = true;
        } else {

            VecInt seqAuxiliary = new VecInt();
            int vars = solver_.nVars();
            solver_.newVar(solver_.nVars() + lits.size());

            for (int i = 0; i < lits.size() - 1; i++) {
                seqAuxiliary.push(++vars);
            }

            for (int i = 0; i < lits.size(); i++) {
                if (i == 0) {
                    // Uncomment this if you want an EO encoding
//                    if (addClause(new VecInt(new int[]{lits.get(i), -seqAuxiliary.get(i)})))
//                        conflict = true;
                    if (addClause(new VecInt(new int[]{-lits.get(i), seqAuxiliary.get(i)})))
                        conflict = true;
                } else if (i == lits.size() - 1) {
                    if (addClause(new VecInt(new int[]{lits.get(i), seqAuxiliary.get(i - 1)})))
                        conflict = true;
                    if (conflict = addClause(new VecInt(new int[]{-lits.get(i), -seqAuxiliary.get(i - 1)})))
                        conflict = true;
                } else {
                    if (addClause(new VecInt(new int[]{-seqAuxiliary.get(i - 1), seqAuxiliary.get(i)})))
                        conflict = true;
                    if (addClause(new VecInt(new int[]{lits.get(i), -seqAuxiliary.get(i), seqAuxiliary.get(i - 1)})))
                        conflict = true;
                    if (addClause(new VecInt(new int[]{-lits.get(i), seqAuxiliary.get(i)})))
                        conflict = true;
                    if (addClause(new VecInt(new int[]{-lits.get(i), -seqAuxiliary.get(i - 1)})))
                        conflict = true;
                }
            }
        }
        return conflict;
    }

    public int analyzeSATConflict(Constr conflict) {

//        System.out.println("conflict= " + conflict.toString());
//        System.out.println("trail= " + solver_.trail.toString());
        org.sat4j.minisat.core.Pair analysisResult = new org.sat4j.minisat.core.Pair();
        int backjumpLevel = -1;
        int rootLevel = 0;

        if (solver_.decisionLevel() == rootLevel) {
            // unsat
            return backjumpLevel;
        } else {
            int conflictTrailLevel = solver_.trail.size();
            // analyze conflict
            try {
                solver_.analyze(conflict, analysisResult);
            } catch (TimeoutException e) {
                // we should never have a timeout
            }
            assert analysisResult.backtrackLevel < solver_.decisionLevel();
            backjumpLevel = Math.max(analysisResult.backtrackLevel, rootLevel);
            solver_.cancelUntil(backjumpLevel);
            assert solver_.decisionLevel() >= rootLevel
                    && solver_.decisionLevel() >= analysisResult.backtrackLevel;

            if (analysisResult.reason == null) {
                backjumpLevel = -1;
                return backjumpLevel;
            }

            //System.out.println("backjumplevel= " + backjumpLevel);
            assert (analysisResult.reason != null);

            solver_.record(analysisResult.reason);
            analysisResult.reason = null;
        }
        return backjumpLevel;
    }


}
