package org.genesys.decide;

import com.microsoft.z3.BoolExpr;
import org.genesys.language.Grammar;
import org.genesys.language.Production;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.utils.LibUtils;
import org.genesys.utils.SATUtils;
import org.genesys.utils.Z3Utils;
import org.sat4j.core.VecInt;
import org.sat4j.minisat.core.Constr;
import org.sat4j.specs.Lbool;

import java.lang.reflect.Array;
import java.util.*;

/**
 * Created by ruben on 9/11/17.
 */
public class MorpheusSolver implements AbstractSolver<BoolExpr, Pair<Node,Node>> {

    private int ITERATION_LIMIT = Integer.MAX_VALUE;

    private Decider decider_;

    private SATUtils satUtils_;

    private Grammar grammar_;

    private int iterations_ = 0;

    private String treeSketch_ = "";

    private double propagateTime_ = 0.0;
    private double backtrackTime_ = 0.0;
    private double decideTime_ = 0.0;
    private double translateTime_ = 0.0;
    private double learnTime_ = 0.0;
    private double step1Time_ = 0.0;
    private double step2Time_ = 0.0;
    private double step3Time_ = 0.0;

    private double backtrackTime1_ = 0.0;
    private double backtrackTime2_ = 0.0;
    private double backtrackTimeTrail_ = 0.0;
    private double backtrackTimeTrailNeo_ = 0.0;
    private double backtrackTimeSAT_ = 0.0;
    private double backtrackTimeBlock_ = 0.0;
    private double backtrackTimeOther_ = 0.0;

    private double decideHigh_ = 0.0;
    private double decideFirst_ = 0.0;
    private double decideInputs_ = 0.0;

    private int nbGlobalLearnts_ = 0;
    private int nbLocalLearnts_ = 0;
    private ArrayList<Double> statsLearnts_ = new ArrayList<>();

    private List<Pair<Integer,String>> currentSketch_ = new ArrayList<>();

    private HashMap<String,Set<Integer>> assignmentsCache_ = new HashMap<>();

    private VecInt currentSketchClause_ = new VecInt();

    /* Size of the program */
    private int maxLen_ = 4;

    /* Maximum number of children of a node */
    private int maxChildren_ = 0;

    /* Producions from inputs */
    private final List<Production> inputProductions_ = new ArrayList<>();
    private final List<List<Production>> lineProductions_ = new ArrayList<>();

    /* Maps types to productions */
    private Map<String, List<Production>> prodTypes_ = new HashMap<>();

    /* Maps production to symbols */
    private Map<String, Object> prodSymbols_ = new HashMap<>();

    /* Domain of productions */
    private final List<Production> domainProductions_ = new ArrayList<>();
    private final List<Production> domainHigher_ = new ArrayList<>();
    private final List<Production> domainFirst_ = new ArrayList<>();
    private final List<Production> domainOutput_ = new ArrayList<>();
    private final List<Production> domainInput_ = new ArrayList<>();

    private final HashMap<Pair<Integer,Integer>,Integer> map2ktree_ = new HashMap<>();
    private final HashMap<Integer,Integer> mapold2new_ = new HashMap<>();
    private final HashMap<Integer,Integer> mapnew2old_ = new HashMap<>();

    private final HashMap<String,List<Integer>> higherGrouping_ = new HashMap<>();

    private List<Node> loc_ = new ArrayList<Node>();

    private List<Integer> sketchNodes_ = new ArrayList<>();

    private HashMap<String, Boolean> cacheAST_ = new HashMap<>();
    private HashMap<String, Boolean> sketches_ = new HashMap<>();

    private HashSet<String> cacheCore_ = new HashSet<>();

    private HashMap<Integer,Pair<Integer,String>> intermediateLearning_ = new HashMap<Integer,Pair<Integer,String>>();
    private boolean treeLearning_ = false;

    private boolean init_ = false;

    private int nodeId_ = 1;

    private int step_ = 1;
    private int step2lvl_ = 1;

    private int learntLine_ = 0;
    private int currentLine_ = 0;

    private final Map<Pair<Integer, Production>, Integer> varNodes_ = new HashMap<>();
    private final Map<Pair<Integer, String>, Integer> nameNodes_ = new HashMap<>();

    private int nbVariables_ = 0;

    private List<Node> nodes_ = new ArrayList<>();

    private List<Pair<Node, Integer>> highTrail_ = new ArrayList<>();
    private final List<List<Pair<Node, Integer>>> trail_ = new ArrayList<>();
    private final List<Pair<Node,Pair<Integer,Integer>>> trailNeo_ = new ArrayList<>();
    private final VecInt trailSAT_ = new VecInt();

    private final VecInt cpTrailSAT_ = new VecInt();

    //private final List<Integer> trailSAT_ = new ArrayList<>();
    private List<Integer> currentSATLevel_ = new ArrayList<>();

    private final List<List<Integer>> backtrack_ = new ArrayList<>();

    /* String to production */
    private Map<String, Production> prodName_ = new HashMap<>();

    private int level_ = 0;
    private int currentChild_ = 0;

    private boolean partial_ = true;

    private Pair<Node,Node> ast_ = null;
    private List<Pair<Integer,Integer>> blockLearn_ = new ArrayList<>();
    private boolean blockLearnFlag_ = false;

    private Node blockAst_ = null;
    private Node learntAst_ = null;

    private VecInt clauseLearn_ = new VecInt();

    private boolean learning_ = false;

    Set<String> binaryComponent_ = new HashSet<String>();

    public MorpheusSolver(Grammar g, Decider decider) {
        satUtils_ = SATUtils.getInstance();
        grammar_ = g;
        decider_ = decider;
        Object start = grammar_.start();
        binaryComponent_.add("ZIPWITH-PLUS");
        binaryComponent_.add("ZIPWITH-MINUS");
        binaryComponent_.add("ZIPWITH-MUL");
        binaryComponent_.add("ZIPWITH-MIN");
        binaryComponent_.add("ZIPWITH-MAX");
        binaryComponent_.add("TAKE");
        binaryComponent_.add("DROP");
        binaryComponent_.add("ACCESS");
        binaryComponent_.add("inner_join");

        statsLearnts_.add(0.0); // global
        statsLearnts_.add(0.0); // local
        statsLearnts_.add(0.0); // avg size of global
        statsLearnts_.add(0.0); // avg size of local
        statsLearnts_.add(0.0); // units
        statsLearnts_.add(0.0); // binary
        statsLearnts_.add(0.0); // ternary

    }

    public MorpheusSolver(Grammar g, int depth, Decider decider, boolean learning) {
        satUtils_ = SATUtils.getInstance();
        maxLen_ = depth;
        grammar_ = g;
        decider_ = decider;
        Object start = grammar_.start();
        for (int i  = 0 ; i < maxLen_; i++){
            trail_.add(new ArrayList<>());
            backtrack_.add(new ArrayList<>());
            lineProductions_.add(new ArrayList<>());
        }
        learning_ = learning;
        binaryComponent_.add("ZIPWITH-PLUS");
        binaryComponent_.add("ZIPWITH-MINUS");
        binaryComponent_.add("ZIPWITH-MUL");
        binaryComponent_.add("ZIPWITH-MIN");
        binaryComponent_.add("ZIPWITH-MAX");
        binaryComponent_.add("TAKE");
        binaryComponent_.add("DROP");
        binaryComponent_.add("ACCESS");
        binaryComponent_.add("inner_join");

        statsLearnts_.add(0.0); // global
        statsLearnts_.add(0.0); // local
        statsLearnts_.add(0.0); // avg size of global
        statsLearnts_.add(0.0); // avg size of local
        statsLearnts_.add(0.0); // units
        statsLearnts_.add(0.0); // binary
        statsLearnts_.add(0.0); // ternary

    }


    @Override
    public Pair<Node,Node> getModel(BoolExpr core, boolean block) {
        if (!init_) {
            init_ = true;
            loadGrammar();
            initDataStructures();
        } else {

            boolean conflict = false;

            if (block || !partial_) {
                if (step_ == 4) {
                    assert (core == null);
                    conflict &= blockModelNeo();
                    //conflict &= blockModel();
                }
                else conflict &= blockModel();
                
                if (blockLearnFlag_) {
                    // I need to learn a clause that blocks the previous ast up to currentLine
                    conflict &= satUtils_.addClause(clauseLearn_, SATUtils.ClauseType.ASSIGNMENT);
                    blockLearnFlag_ = false;
                    if (conflict)
                        return null;
                }
                if (conflict) {
                    return null;
                }
            } else {
                if (blockLearnFlag_) {
                    backtrackStep1(0,false);
                    // I need to learn a clause that blocks the previous ast up to currentLine
                    conflict &= satUtils_.addClause(clauseLearn_, SATUtils.ClauseType.ASSIGNMENT);
                    blockLearnFlag_ = false;
                    if (conflict)
                        return null;
                    step_ = 1;
                }

                if (step_ == 4 && partial_){
                    // continue the search
                    step_ = 3;
                }
            }
            partial_ = true;
        }

        Pair<Node,Node> node = search();
        return node;
    }

    public ArrayList<Double> getLearnStats(){
        return statsLearnts_;
    }

    public boolean learnCoreSet(List<List<Pair<Integer, List<String>>>> core, boolean global) {

        boolean confl = false;
        for (List<Pair<Integer, List<String>>> s : core){
            confl = confl & learnCore(s, global);
        }

        return confl;
    }

    public boolean learnCore(List<Pair<Integer, List<String>>> core, boolean global) {
        //long s = LibUtils.tick();
        boolean conflict = false;
        //System.out.println("core = " + core);

        HashMap<Integer,String> node2function = new HashMap<>();
        List<Node> bfs = new ArrayList<>();
        Node root = ast_.t0;
        bfs.add(root);
        while (!bfs.isEmpty()) {
            Node node = bfs.remove(bfs.size() - 1);
            //assert (mapnew2old_.containsKey(node.id));
            //int node_id = mapnew2old_.get(node.id);
            int node_id = node.id;
            node2function.put(node_id,node.function);
            for (int i = 0; i < node.children.size(); i++)
                bfs.add(node.children.get(i));
        }

        List<List<Integer>> eqClauses = new ArrayList<>();
        Set<Integer> seen = new HashSet<>();
        String learnt = "";
        boolean exists = false;

        List<Pair<Integer, List<String>>> debug_core = new ArrayList<>();

        for (Pair<Integer,List<String>> p : core){
            List<Integer> eq = new ArrayList<>();
            List<Integer> eq2 = new ArrayList<>();
            Pair pp = null;
            assert (mapnew2old_.containsKey(p.t0)) : core;
            int node_id = mapnew2old_.get(p.t0);
            if (seen.contains(node_id))
                continue;
            debug_core.add(p);
            learnt = learnt + "[(" + node_id + ") ";
            seen.add(node_id);
            for (String l : p.t1) {
                Pair<Integer, String> id2 = new Pair<>(node_id, l);
                if (!nameNodes_.containsKey(id2))
                    continue;
                eq.add(nameNodes_.get(id2));
                learnt = learnt + l + " ";
                if (l.equals("ZIPWITH-PLUS") ||
                        l.equals("ZIPWITH-MINUS") ||
                        l.equals("ZIPWITH-MUL") ||
                        l.equals("ZIPWITH-MIN") ||
                        l.equals("ZIPWITH-MAX") ||
                        l.equals("TAKE") ||
                        l.equals("DROP") ||
                        l.equals("ACCESS") ||
                        l.equals("inner_join")
                        ){
                    exists = true;
                }

                if (treeLearning_ && global && exists){
                    if (intermediateLearning_.containsKey(node_id)){
                        pp = intermediateLearning_.get(node_id);
                        assert (nameNodes_.containsKey(pp));
                        assert (pp.t1.toString().startsWith("line"));
                        eq2.add(nameNodes_.get(pp));
                        Pair<Integer,List<String>> rr = new Pair<Integer,List<String>>((Integer)pp.t0,new ArrayList<String>());
                        rr.t1.add((String)pp.t1);
                        debug_core.add(rr);
                        mapnew2old_.put((Integer)pp.t0,(Integer)pp.t0);
                    }
                }
            }
            //}
            learnt = learnt + "]";
            if (!eq.isEmpty()) {
                eqClauses.add(eq);
                if (!eq2.isEmpty()){
                    eqClauses.add(eq2);
                    learnt = learnt + "[(" + pp.t0 + ") " + pp.t1 + " ]";
                }
            }

        }

        if (!eqClauses.isEmpty()) {

            if (!cacheCore_.contains(debug_core.toString())) {
                cacheCore_.add(debug_core.toString());
                if (core.size() <= 3 || global) conflict = learnCoreSimple(debug_core);
                else {
                    conflict = learnCoreLocal(debug_core, learntLine_);
                }
            }
            //else conflict = SATUtils.getInstance().learnCoreLocal(eqClauses, learntLine_);

        }

//        long e = LibUtils.tick();
//        learnTime_ += LibUtils.computeTime(s,e);
        return conflict;

    }

    private void GeneratePermutations(List<List<String>> Lists, List<List<String>> result, int depth, List<String> current) {
        if(depth == Lists.size())
        {
            result.add(current);
            return;
        }

        for(int i = 0; i < Lists.get(depth).size(); ++i)
        {
            List c = new ArrayList();
            c.addAll(current);
            c.add(Lists.get(depth).get(i));
            GeneratePermutations(Lists, result, depth + 1, c);
        }
    }

    public boolean learnCoreLocal(List<Pair<Integer, List<String>>> core, int line) {

        boolean conflict = false;

        List<Integer> nodes = new ArrayList<>();
        List<List<String>> s = new ArrayList<>();
        for (Pair<Integer, List<String>> p : core) {
            s.add(p.t1);
            nodes.add(p.t0);
        }

        List<List<String>> result = new ArrayList<>();
        List<String> current = new ArrayList<>();

        List<Pair<VecInt, List<Pair<Integer, String>>>> clauses = new ArrayList<>();

        GeneratePermutations(s, result, 0, current);
        for (List<String> r : result) {
            Pair<VecInt, List<Pair<Integer, String>>> clause = new Pair<>(new VecInt(), new ArrayList<>());
            boolean root = true;
            boolean ignore = false;
            for (int i = 0; i < r.size(); i++) {
                assert (mapnew2old_.containsKey(nodes.get(i)));
                int node_id = mapnew2old_.get(nodes.get(i));
                Pair<Integer, String> id = new Pair<>(node_id, r.get(i));
                // FIXME: Yu is giving me constants that do not exist
                if (!nameNodes_.containsKey(id)) {
                    ignore = true;
                    break;
                    //continue;
                }
                assert (nameNodes_.containsKey(id));
                clause.t0.push(-nameNodes_.get(id));
                clause.t1.add(id);
            }

            if (!ignore) {

                statsLearnts_.set(1,statsLearnts_.get(1)+1);
                double avg = (nbLocalLearnts_ * statsLearnts_.get(3) + clause.t0.size())/(nbLocalLearnts_+1);
                statsLearnts_.set(3,avg);
                nbLocalLearnts_++;

                conflict &= SATUtils.getInstance().addLearnt(clause.t0, line);
            }
        }

        return conflict;
    }

    public boolean learnCoreSimple(List<Pair<Integer, List<String>>> core) {

        boolean conflict = false;

        List<Integer> nodes = new ArrayList<>();
        List<List<String>> s = new ArrayList<>();
        for (Pair<Integer,List<String>> p : core) {
            s.add(p.t1);
            nodes.add(p.t0);
        }

        List<List<String>> result = new ArrayList<>();
        List<String> current = new ArrayList<>();

        List<Pair<VecInt, List<Pair<Integer,String>>>> clauses = new ArrayList<>();

        GeneratePermutations(s,result,0, current);
        for (List<String> r : result){
            Pair<VecInt, List<Pair<Integer,String>>> clause = new Pair<>(new VecInt(),new ArrayList<>());
            boolean root = true;
            for (int i = 0;  i < r.size(); i++){
                assert (mapnew2old_.containsKey(nodes.get(i)));
                int node_id = mapnew2old_.get(nodes.get(i));
                Pair<Integer, String> id = new Pair<>(node_id, r.get(i));
//                if (nodes.get(i) == 0)
//                    root = true;
//                if (ast_.t0.children.get(0).id == node_id)
//                    root = true;
                assert (nameNodes_.containsKey(id));
                clause.t0.push(-nameNodes_.get(id));
                clause.t1.add(id);
            }

            //SATUtils.getInstance().addClause(clause.t0, SATUtils.ClauseType.GLOBAL);
            //clauses.add(clause);

            if (clause.t0.size() == 1)
                statsLearnts_.set(4,statsLearnts_.get(4)+1);
            else if (clause.t0.size() == 2)
                statsLearnts_.set(5,statsLearnts_.get(5)+1);
            else if (clause.t0.size() == 3)
                statsLearnts_.set(6,statsLearnts_.get(6)+1);

            if (clause.t0.size() <= 2){

                statsLearnts_.set(0,statsLearnts_.get(0)+1);
                double avg = (nbGlobalLearnts_ * statsLearnts_.get(2) + clause.t0.size())/(nbGlobalLearnts_+1);
                statsLearnts_.set(2,avg);
                nbGlobalLearnts_++;

                conflict &= SATUtils.getInstance().addClause(clause.t0, SATUtils.ClauseType.GLOBAL);
            } else {

                statsLearnts_.set(0,statsLearnts_.get(0)+1);
                double avg = (nbGlobalLearnts_ * statsLearnts_.get(2) + clause.t0.size())/(nbGlobalLearnts_+1);
                statsLearnts_.set(2,avg);
                nbGlobalLearnts_++;

                conflict &= SATUtils.getInstance().addClause(clause.t0, SATUtils.ClauseType.LOCAL);
                //clauses.add(clause);
            }

            /*

            //root = false;
            if (clause.t0.size() <= 3)
                root = false;

            if (!root){
                //System.out.println("learning = " + clause.t1);
                SATUtils.getInstance().addClause(clause.t0, SATUtils.ClauseType.GLOBAL);
            } else {

                if (!clause.t0.contains(cpTrailSAT_.get(cpTrailSAT_.size() - 1))) {
                    // partial assignment
                    boolean contains = true;
                    for (Pair<Integer, String> pair : clause.t1) {
                        if (sketchNodes_.contains(pair.t0) && !currentSketch_.contains(pair)) {
                            contains = false;
                            break;
                        }
                    }
                    if (contains) {
                        // learnt clause is relevant to this sketch
                        // find the relevant part of the trail
                        VecInt cc = new VecInt();
                        int pos = 0;
                        for (int i = cpTrailSAT_.size() - 1; i >= 0; i--) {
                            if (clause.t0.contains(cpTrailSAT_.get(i))) {
                                pos = i;
                                break;
                            }
                        }

                        for (int i = 0; i < pos; i++) {
                            cc.push(cpTrailSAT_.get(i));
                        }

                        if (!assignmentsCache_.containsKey(cc.toString())) {
                            assignmentsCache_.put(cc.toString(), new HashSet());
                        }
                        assignmentsCache_.get(cc.toString()).add(cpTrailSAT_.get(pos));
                        // maybe it is also relevant to other eq classes
                        clauses.add(clause);
                    } else {
                        clauses.add(clause);
                    }
                } else {
                    clauses.add(clause);
                }
            }
            */
        }

//        if (!clauses.isEmpty())
//            SATUtils.getInstance().updateEqLearnts(clauses);

        return conflict;
    }


    public Pair<Node,Node> getCoreModel(List<Pair<Integer, List<String>>> core, boolean block, boolean global) {
        if (!init_) {
            init_ = true;
            loadGrammar();
            initDataStructures();
        } else {
            boolean conflict = false;
            //if (step_ == 4) conflict = blockModelNeo();
            //else
            conflict = blockModel();

            if (blockLearnFlag_) {
                // I need to learn a clause that blocks the previous ast up to currentLine
                conflict &= satUtils_.addClause(clauseLearn_, SATUtils.ClauseType.ASSIGNMENT);
                blockLearnFlag_ = false;
            }

            if (conflict) {
                return null;
            }
            else {
                boolean confl = learnCore(core, global);
                if (confl){
                    System.out.println("s UNSATISFIABLE - learning core");
                    return null;
                }
            }
            partial_ = true;
        }

        Pair<Node,Node> node = search();
        return node;
    }

    public Pair<Node,Node> getCoreModelSet(List<List<Pair<Integer, List<String>>>> core, boolean block, boolean global) {
        if (!init_) {
            init_ = true;
            loadGrammar();
            initDataStructures();
        } else {
            boolean conflict = false;
            //if (step_ == 4) conflict = blockModelNeo();
            conflict = blockModel();

            if (blockLearnFlag_) {
                // I need to learn a clause that blocks the previous ast up to currentLine
                conflict &= satUtils_.addClause(clauseLearn_, SATUtils.ClauseType.ASSIGNMENT);
                blockLearnFlag_ = false;
            }

            if (conflict) {
                return null;
            }
            else {
                boolean confl = learnCoreSet(core, global);
                if (confl){
                    System.out.println("s UNSATISFIABLE - learning core");
                    return null;
                }
            }
            partial_ = true;
        }

        Pair<Node,Node> node = search();
        return node;
    }

    private void buildTree(){

        Node node = new Node();
        int id = 1;
        node.id = id++;
        node.level = 1;
        map2ktree_.put(new Pair<Integer,Integer>(1,0),1);

        List<Node> working = new ArrayList<>();
        working.add(node);
        while (!working.isEmpty()){
            node = working.remove(working.size() - 1);
            for (int i = 0; i < maxChildren_; i++){
                Node child = new Node();
                child.id = id++;
                child.level = node.level+1;
                node.addChild(child);
                map2ktree_.put(new Pair<Integer,Integer>(i+1,node.id),child.id);
                if (child.level <= loc_.size())
                    working.add(child);
            }
        }

    }


    @Override
    public boolean isPartial(){
        return partial_;
    }

    private Node createNode(List<Production> root, List<Production> children) {

        Node node = new Node("", new ArrayList<>(), root);
        node.id = nodeId_++;
        nodes_.add(node);
        for (int i = 0; i < maxChildren_; i++) {
            Node child = new Node("", new ArrayList<>(), children);
            child.id = nodeId_++;
            nodes_.add(child);
            node.addChild(child);
        }
        return node;
    }


    private void initDataStructures() {

        // Each line has its own subtree
        for (int i = 0; i < maxLen_; i++){
            Node node = null;

            List<Production> domain = new ArrayList<>();
            domain.addAll(domainFirst_);
            for (int j = i-1; j >= 0; j--)
                domain.addAll(lineProductions_.get(j));

            if (i == 0) node = createNode(domainInput_,domain);
            else if (i == maxLen_-1) node = createNode(domainOutput_,domain);
            else node = createNode(domainHigher_,domain);

            createVariables(node);

            loc_.add(node);
            highTrail_.add(new Pair<Node, Integer>(node,highTrail_.size()+1));
            sketchNodes_.add(highTrail_.get(i).t0.id);
        }

        // Create empty SAT solver
        satUtils_.createSolver();
        satUtils_.createVars(nbVariables_);

        buildSATFormula();
        buildTree();

        // Decision level for Neo and SAT solver
        level_ = 0;
        currentLine_ = 0;
        currentSATLevel_.add(0);

    }

    private void buildSATFormula() {

        boolean conflict = false;

        // If a production is used in a parent node then this implies restrictions on the children
        for (Node node : nodes_) {
            for (Production p : node.domain) {

                int productionVar = varNodes_.get(new Pair<Integer, Production>(node.id, p));
                boolean exist_production = true;
                for (int i = 0; i < p.inputs.length; i++){
                    if(!prodTypes_.containsKey(p.inputs[i].toString())) {
                        exist_production = false;
                        break;
                    }
                }

                if (!exist_production) {
                    continue;
                }

                for (int i = 0; i < p.inputs.length; i++) {
                    ArrayList<Production> occurs = new ArrayList<>();
                    VecInt clause = new VecInt();
                    clause.push(-productionVar);
                    for (Production pc : prodTypes_.get(p.inputs[i].toString())) {
                        if (node.children.get(i).domain.contains(pc)) {
                            Pair<Integer, Production> pair = new Pair<Integer, Production>(node.children.get(i).id, pc);
                            assert (varNodes_.containsKey(pair));
                            // Parent restricts the domain of the child (positively)
                            clause.push(varNodes_.get(pair));
                            occurs.add(pc);
                        }
                    }
                    if (clause.size() > 1) {
                        conflict = satUtils_.addClause(clause);
                        assert(!conflict);
                    }

                    for (Production pc : node.children.get(i).domain) {
                        if (!occurs.contains(pc)) {
                            VecInt lits = new VecInt(new int[]{-productionVar, -varNodes_.get(new Pair<Integer, Production>(node.children.get(i).id, pc))});
                            // Parent restricts the domain of child (negatively)
                            conflict = satUtils_.addClause(lits);
                            assert(!conflict);
                        }
                    }
                }

                // If this node contains less than k children then the remaining will be empty
                if (!node.children.isEmpty()) {
                    for (int i = p.inputs.length; i < maxChildren_; i++) {
                        for (Production pc : node.children.get(i).domain) {
                            VecInt lits = new VecInt(new int[]{-productionVar, -varNodes_.get(new Pair<Integer, Production>(node.children.get(i).id, pc))});
                            conflict = satUtils_.addClause(lits);
                            assert(!conflict);
                        }
                    }
                }
            }
        }

//        /* Domain specific constraints for R */
//        // group_by can only be a child of summarise
        if (prodName_.containsKey("summarise") && prodName_.containsKey("group_by")) {

            for (int i = 0; i < maxLen_-1; i++) {
                Node parent = highTrail_.get(i).t0;
                Production g = prodName_.get("group_by");
                Node child = highTrail_.get(i + 1).t0;
                Production s = prodName_.get("summarise");
                int v1 = varNodes_.get(new Pair<Integer, Production>(parent.id, g));
                int v2 = varNodes_.get(new Pair<Integer, Production>(child.id, s));
                assert (lineProductions_.get(i).size()==1);
                Production itm = lineProductions_.get(i).get(0);
                int v3 = varNodes_.get(new Pair<Integer, Production>(child.children.get(0).id, itm));
                VecInt lits = new VecInt(new int[]{-v1, v2});
                conflict = satUtils_.addClause(lits);
                assert(!conflict);
                VecInt lits2 = new VecInt(new int[]{-v1, v3});
                conflict = satUtils_.addClause(lits2);
                assert(!conflict);
            }

            // group_by cannot be at the root level
            Node root = highTrail_.get(highTrail_.size()-1).t0;
            Production gg = prodName_.get("group_by");
            int var = varNodes_.get(new Pair<Integer, Production>(root.id, gg));
            VecInt lits = new VecInt(new int[]{-var});
            conflict = satUtils_.addClause(lits);
            assert(!conflict);
        }

        if (prodName_.containsKey("filter") && prodName_.containsKey("select")){
            // Filter select is equivalent to select filter
            // Only allow one of them to happen
            for (int i = 0; i < highTrail_.size()-1; i++){
                Production f = prodName_.get("filter");
                Production s = prodName_.get("select");
                Node node = highTrail_.get(i).t0;
                Node next = highTrail_.get(i+1).t0;
                int v1 = varNodes_.get(new Pair<Integer, Production>(node.id, s));
                int v2 = varNodes_.get(new Pair<Integer, Production>(next.id, f));
                VecInt clause = new VecInt(new int[]{-v1,-v2});
                conflict = satUtils_.addClause(clause);
                assert(!conflict);

                assert (lineProductions_.get(i).size()==1);
                Production itm = lineProductions_.get(i).get(0);
                int v3 = varNodes_.get(new Pair<Integer, Production>(next.children.get(0).id, itm));
                VecInt lits2 = new VecInt(new int[]{-v1, v3});
                conflict = satUtils_.addClause(lits2);
                assert(!conflict);
            }
        }

        if (prodName_.containsKey("mutate")){
            // At most one mutate
            VecInt clause = new VecInt();
            for (int i = 0; i < highTrail_.size(); i++){
                Production p = prodName_.get("mutate");
                Node node = highTrail_.get(i).t0;
                int var = varNodes_.get(new Pair<Integer,Production>(node.id, p));
                clause.push(var);
            }
            conflict = satUtils_.addAMK(clause, 1);
            assert(!conflict);
        }

        if (prodName_.containsKey("inner_join")){
            // At most one mutate
            VecInt clause = new VecInt();
            for (int i = 0; i < highTrail_.size(); i++){
                Production p = prodName_.get("inner_join");
                Node node = highTrail_.get(i).t0;
                int var = varNodes_.get(new Pair<Integer,Production>(node.id, p));
                clause.push(var);
            }
            conflict = satUtils_.addAMK(clause, 1);
            assert(!conflict);
        }

        if (prodName_.containsKey("filter")) {
            VecInt clause = new VecInt();
            for (int i = 0; i < highTrail_.size(); i++) {
                Production p = prodName_.get("filter");
                Node node = highTrail_.get(i).t0;
                int var = varNodes_.get(new Pair<Integer, Production>(node.id, p));
                clause.push(var);
            }
            conflict = satUtils_.addAMK(clause, 1);
            assert(!conflict);
        }

        /* Domain specific constraints for DeepCoder */

        String[] amo = {"ACCESS", "MAXIMUM", "COUNT", "MINIMUM", "SUM", "HEAD", "LAST", "FILTER", "SORT", "REVERSE", "TAKE", "DROP"};
        String[] map = {"MAP-MUL","MAP-DIV","MAP-PLUS","MAP-POW"};
        String[] zipwith = {"ZIPWITH-PLUS","ZIPWITH-MINUS","ZIPWITH-MUL","ZIPWITH-MIN","ZIPWITH-MAX"};
        String[] scanl1 = {"SCANL1-PLUS","SCANL1-MINUS","SCANL1-MUL","SCANL1-MIN","SCANL1-MAX"};

        for (String s : amo){
            if (prodName_.containsKey(s)) {
                VecInt clause = new VecInt();
                for (int i = 0; i < highTrail_.size(); i++) {
                    Production p = prodName_.get(s);
                    Node node = highTrail_.get(i).t0;
                    Pair<Integer,Production> pp = new Pair<Integer, Production>(node.id, p);
                    if (varNodes_.containsKey(pp)) {
                        int var = varNodes_.get(pp);
                        clause.push(var);
                    }
                }
                if (clause.size() > 1) {
                    conflict = satUtils_.addAMK(clause, 1);
                    assert (!conflict);
                }
            }
        }

        VecInt map_clause = new VecInt();
        for (String s : map){
            if (prodName_.containsKey(s)) {
                for (int i = 0; i < highTrail_.size(); i++) {
                    Production p = prodName_.get(s);
                    Node node = highTrail_.get(i).t0;
                    Pair<Integer,Production> pp = new Pair<Integer, Production>(node.id, p);
                    if (varNodes_.containsKey(pp)) {
                        int var = varNodes_.get(pp);
                        map_clause.push(var);
                    }
                }
            }
        }
        if (map_clause.size() > 2){
            conflict = satUtils_.addAMK(map_clause, 2);
            assert (!conflict);
        }

        VecInt zipwith_clause = new VecInt();
        for (String s : map){
            if (prodName_.containsKey(s)) {
                for (int i = 0; i < highTrail_.size(); i++) {
                    Production p = prodName_.get(s);
                    Node node = highTrail_.get(i).t0;
                    Pair<Integer,Production> pp = new Pair<Integer, Production>(node.id, p);
                    if (varNodes_.containsKey(pp)) {
                        int var = varNodes_.get(pp);
                        zipwith_clause.push(var);
                    }
                }
            }
        }
        if (zipwith_clause.size() > 2){
            conflict = satUtils_.addAMK(zipwith_clause, 2);
            assert (!conflict);
        }

        VecInt scanl1_clause = new VecInt();
        for (String s : map){
            if (prodName_.containsKey(s)) {
                for (int i = 0; i < highTrail_.size(); i++) {
                    Production p = prodName_.get(s);
                    Node node = highTrail_.get(i).t0;
                    Pair<Integer,Production> pp = new Pair<Integer, Production>(node.id, p);
                    if (varNodes_.containsKey(pp)) {
                        int var = varNodes_.get(pp);
                        scanl1_clause.push(var);
                    }
                }
            }
        }
        if (scanl1_clause.size() > 2){
            conflict = satUtils_.addAMK(scanl1_clause, 2);
            assert (!conflict);
        }

//        if (prodName_.containsKey("ACCESS")){
//            // ACCESS cannot be the first line
//            Node root = highTrail_.get(0).t0;
//            Production gg = prodName_.get("ACCESS");
//            if (varNodes_.containsKey(new Pair<Integer, Production>(root.id, gg))) {
//                int var = varNodes_.get(new Pair<Integer, Production>(root.id, gg));
//                VecInt lits = new VecInt(new int[]{-var});
//                satUtils_.addClause(lits);
//            }
//        }

        // FIXME: At most one variable is assigned at each node -- this has some issue
        for (Node node : nodes_) {
            VecInt clause = new VecInt();
            for (Production p : node.domain){
                assert (varNodes_.containsKey(new Pair<Integer, Production>(node.id, p)));
                int productionVar = varNodes_.get(new Pair<Integer, Production>(node.id, p));
                if (p.function.startsWith("input") || p.function.startsWith("line"))
                    clause.push(productionVar);
            }
            if (clause.size() > 0){
                // only consider inputs
                conflict = satUtils_.addAMK(clause, 1);
                assert (!conflict);
            }
            //conflict = satUtils_.addAMK(clause,1);
            //assert (!conflict);
        }

        Iterator it = higherGrouping_.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();
            VecInt amk = new VecInt();
            for (int i = 0 ;i < ((List)pair.getValue()).size(); i++){
                amk.push(((List<Integer>)pair.getValue()).get(i));

                // disabled for DeepCoder -- maybe add it as preference for Morpheus?
//                if (i != ((List)pair.getValue()).size()-1){
//                    // There are no two consecutive higher order function calls
//                    VecInt clause = new VecInt(new int[]{-((List<Integer>)pair.getValue()).get(i),-((List<Integer>)pair.getValue()).get(i+1)});
//                    satUtils_.addClause(clause);
//                }
            }
            // At most 2 occurrences of each higher order component in the sketch
            conflict = satUtils_.addAMK(amk, 2);
            assert(!conflict);
            it.remove();
        }

        // At least one input must be used in the first line of code

        VecInt clause_input = new VecInt();
        for (Node node : highTrail_.get(0).t0.children){

            for (Production p : inputProductions_){
                int productionVar = varNodes_.get(new Pair<Integer, Production>(node.id, p));
                clause_input.push(productionVar);
            }
        }
        conflict = satUtils_.addClause(clause_input);
        assert(!conflict);

        // The intermediate results cannot be used before they are created
        // ---> This is encoded in the domain


        // An intermediate result is used exactly once (do not allow let binding)
        List<VecInt> letbind = new ArrayList<>();
        for (int i = 0; i < maxLen_; i++)
            letbind.add(new VecInt());

        for (int i = 0 ; i < maxLen_; i++){
            for (int j = i-1; j >= 0; j--){
                for (Production p : lineProductions_.get(j)){
                    for (Node node : highTrail_.get(i).t0.children) {
                        int productionVar = varNodes_.get(new Pair<Integer, Production>(node.id, p));
                        letbind.get(j).push(productionVar);
                    }
                }
            }
        }

        for (int i = 0; i < maxLen_; i++){
            if (!letbind.get(i).isEmpty()) {
                conflict = satUtils_.addClause(letbind.get(i));
                assert(!conflict);
                conflict = satUtils_.addAMK(letbind.get(i),1);
                assert(!conflict);
            }
        }

        // Every input is used at least once and at most twice
        List<List<Integer>> inputs_used = new ArrayList<List<Integer>>();
        for (int i = 0 ; i < inputProductions_.size(); i++)
            inputs_used.add(new ArrayList<>());

        for (int i = 0; i < maxLen_; i++) {
            for (Production p : inputProductions_){
                for (Node node : highTrail_.get(i).t0.children) {
                    assert (varNodes_.containsKey(new Pair<Integer, Production>(node.id, p)));
                    int var = varNodes_.get(new Pair<Integer, Production>(node.id, p));
                    String[] parts = p.function.split("input");
                    assert (parts.length == 2);
                    inputs_used.get(Integer.valueOf(parts[1])).add(var);
                }
            }
        }

        for (int i = 0; i < inputs_used.size(); i++){
            VecInt clause = new VecInt();
            for (int j = 0; j < inputs_used.get(i).size(); j++){
                clause.push(inputs_used.get(i).get(j));
            }
            conflict = satUtils_.addClause(clause); // at least once
            assert(!conflict);
            conflict = satUtils_.addAMK(clause, 2); // at most twice
            assert(!conflict);
            //satUtils_.addEO(clause, 1);
        }

        // If an intermediate result has type T then it cannot have type T'
        // ---> is a consequence of the previous statement when we do not allow let binding


        // If an intermediate result if of type T then the root has to be a production rule of that type
        for (int i = 0; i < maxLen_-1; i++) {

            if (i == 0) {
                for (Production p : domainInput_) {
                    Node node = highTrail_.get(i).t0;
                    int productionVar = varNodes_.get(new Pair<Integer, Production>(node.id, p));

                    for (int j = maxLen_-1; j > i; j--){
                        for (Node n : highTrail_.get(j).t0.children) {
                            for (Production l : lineProductions_.get(i)) {
                                assert (varNodes_.containsKey(new Pair<Integer, Production>(n.id, l)));
                                int lineVar = varNodes_.get(new Pair<Integer, Production>(n.id, l));
                                if (!p.source.equals(l.source)) {
                                    VecInt clause = new VecInt(new int[]{-productionVar, -lineVar});
                                    conflict = satUtils_.addClause(clause);
                                    assert(!conflict);
                                }
                            }
                        }
                    }
                }
            } else {
                for (Production p : domainHigher_) {
                    Node node = highTrail_.get(i).t0;
                    int productionVar = varNodes_.get(new Pair<Integer, Production>(node.id, p));

                    for (int j = maxLen_-1; j > i; j--){
                        for (Node n : highTrail_.get(j).t0.children) {
                            for (Production l : lineProductions_.get(i)) {
                                assert (varNodes_.containsKey(new Pair<Integer, Production>(n.id, l)));
                                int lineVar = varNodes_.get(new Pair<Integer, Production>(n.id, l));
                                if (!p.source.equals(l.source)) {
                                    VecInt clause = new VecInt(new int[]{-productionVar, -lineVar});
                                    conflict = satUtils_.addClause(clause);
                                    assert(!conflict);
                                }
                            }
                        }
                    }
                }
            }
        }

        //long s = LibUtils.tick();
        Constr conf = satUtils_.propagate();
//        long e = LibUtils.tick();
//        propagateTime_ += LibUtils.computeTime(s,e);
        assert (conf == null);
    }

    private <T> void loadGrammar() {
        List<Production<T>> prods = grammar_.getProductions();
        prods.addAll(grammar_.getInputProductions());
        //prods.addAll(grammar_.getLineProductions(maxLen_));

        inputProductions_.addAll((List<Production<T>>) grammar_.getInputProductions());

        for (Production<T> prod : (List<Production<T>>) grammar_.getLineProductions(maxLen_)){
            for (int i = 0 ; i < maxLen_; i++){
                if (prod.function.startsWith("line" + Integer.toString(i))){
                    lineProductions_.get(i).add(prod);
                }
            }
        }

        prods.addAll(grammar_.getLineProductions(maxLen_));

        for (Production<T> prod : prods) {

            if (!prodTypes_.containsKey(prod.source.toString())) {
                prodTypes_.put(prod.source.toString(), new ArrayList<Production>());
            }

            prodTypes_.get(prod.source.toString()).add(prod);

            prodSymbols_.put(prod.function, prod.source);

            prodName_.put(prod.function, prod);

            if (prod.inputs.length > maxChildren_)
                maxChildren_ = prod.inputs.length;

            if (prod.function.startsWith("line"))
                continue;

            domainProductions_.add(prod);

            if (prod.higher) {
                domainHigher_.add(prod);

                if (grammar_.getOutputType().toString().equals(prod.source.toString())) {
                    domainOutput_.add(prod);
                }
                //domainOutput_.add(prod);

                // FIXME: we could potentially prune some input productions
                domainInput_.add(prod);

            } else {
                domainFirst_.add(prod);
            }

        }
    }

    private <T> void createVariables(Node node) {

        for (Production p : node.domain) {
            Pair<Integer, Production> pair = new Pair<Integer, Production>(node.id, p);
            Pair<Integer, String> pair2 = new Pair<Integer, String>(node.id, p.function);
            varNodes_.put(pair, ++nbVariables_);
            nameNodes_.put(pair2,nbVariables_);
//            System.out.println("pair2 = " + pair2);
            if (p.higher) {
                if (!higherGrouping_.containsKey(p.function)) {
                    higherGrouping_.put(p.function, new ArrayList<>());
                    higherGrouping_.get(p.function).add(nbVariables_);
                } else higherGrouping_.get(p.function).add(nbVariables_);
            }

        }

        for (int i = 0; i < node.children.size(); i++)
            createVariables(node.children.get(i));
    }

    public String nextDecisionHigher(List<String> domain){

        List ancestors = new ArrayList<>();
        assert (level_ < highTrail_.size());
        for (int i  = level_-1; i >= 0; i--){
            assert(!highTrail_.get(i).t0.function.equals(""));
            ancestors.add(highTrail_.get(i).t0.function);
        }
        Collections.reverse(ancestors);


        String decision = decider_.decideSketch(ancestors, domain, level_);
        //assert (!decision.equals(""));
        return decision;
    }

    public String nextDecision(List<String> domain) {

//        List ancestors = new ArrayList<>();
//        assert (level_ < highTrail_.size());
//        for (int i  = level_-1; i >= 0; i--){
//            assert(!highTrail_.get(i).t0.function.equals(""));
//            ancestors.add(highTrail_.get(i).t0.function);
//        }
//        Collections.reverse(ancestors);


        String decision = decider_.decide(new ArrayList<>(), domain);
        assert (!decision.equals(""));
        return decision;
    }

    private Node decideInputs() {
        //long s = LibUtils.tick();

        Production decisionNeo = null;
        int decisionSAT = -1;
        int decisionComponent = -1;

        Node node = trail_.get(currentLine_).get(currentChild_).t0;
        Map<String, Pair<Production, Integer>> decideMap = new HashMap<>();
        List<String> decideDomain = new ArrayList<>();
        for (Production p : node.domain) {
            if (p.function.startsWith("input") || p.function.startsWith("line")) {
                int var = varNodes_.get(new Pair<Integer, Production>(node.id, p));

                if (satUtils_.getSolver().truthValue(var) == Lbool.UNDEFINED ||
                        satUtils_.getSolver().truthValue(var) == Lbool.TRUE) {
                    decideMap.put(p.function, new Pair<Production, Integer>(p, var));
                    decideDomain.add(p.function);
                }
            }
        }

        if (!decideDomain.isEmpty()) {
            String decision = nextDecision(decideDomain);
            Pair<Production, Integer> p = decideMap.get(decision);
            decisionNeo = p.t0;
            decisionSAT = p.t1;
            decisionComponent = p.t0.id;
        }

        if (decisionNeo == null) {
//            long e = LibUtils.tick();
//            decideTime_ += LibUtils.computeTime(s,e);
            return null;
        } else {
            node.function = decisionNeo.function;
            node.decision = decisionNeo;
            node.component = decisionComponent;
            node.level = level_;

            //System.out.println("NEO decision Inputs = " + decisionNeo.function + " @" + level_ + " node ID = " + node.id + " SAT decision= " + decisionSAT + " assume= " + satUtils_.posLit(decisionSAT));

            Pair<Integer,Integer> p = new Pair<Integer,Integer>(currentLine_,currentChild_);
            Pair<Node, Pair<Integer,Integer>> p2 = new Pair<Node, Pair<Integer,Integer>>(node, p);

            trailNeo_.add(p2);
            trailSAT_.push(-decisionSAT);
            assert (satUtils_.getSolver().truthValue(decisionSAT) != Lbool.FALSE);
            if (satUtils_.getSolver().truthValue(decisionSAT) == Lbool.UNDEFINED)
                satUtils_.getSolver().assume(satUtils_.posLit(decisionSAT));

            currentSATLevel_.add(satUtils_.getSolver().decisionLevel());

            for (int i = 0; i < decisionNeo.inputs.length; i++) {
                trail_.get(level_).add(new Pair<Node, Integer>(node.children.get(i), level_));
            }
            level_++;

//            long e = LibUtils.tick();
//            decideTime_ += LibUtils.computeTime(s,e);
            return node;
        }
    }


    public Node decideFirst(){
        //long s = LibUtils.tick();

        Production decisionNeo = null;
        int decisionSAT = -1;
        int decisionComponent = -1;

        Node node = trail_.get(currentLine_).get(currentChild_).t0;
        assert (node.equals(highTrail_.get(currentLine_).t0.children.get(currentChild_)));
        Map<String, Pair<Production, Integer>> decideMap = new HashMap<>();
        List<String> decideDomain = new ArrayList<>();
        for (Production p : node.domain) {
            int var = varNodes_.get(new Pair<Integer, Production>(node.id, p));

            if (satUtils_.getSolver().truthValue(var) == Lbool.UNDEFINED ||
                    satUtils_.getSolver().truthValue(var) == Lbool.TRUE) {
                Pair<Production, Integer> pp = new Pair<Production, Integer>(p, var);
                if (assignmentsCache_.containsKey(trailSAT_.toString())) {
                    if (!assignmentsCache_.get(trailSAT_.toString()).contains(-pp.t1)) {
                        decideMap.put(p.function, pp);
                        decideDomain.add(p.function);
                    }
                } else {
                    decideMap.put(p.function, pp);
                    decideDomain.add(p.function);
                }
            }
        }

        if (!decideDomain.isEmpty()) {
            String decision = nextDecision(decideDomain);
            Pair<Production, Integer> p = decideMap.get(decision);
            decisionNeo = p.t0;
            decisionSAT = p.t1;
            decisionComponent = p.t0.id;
        }

        if (decisionNeo == null){
//            long e = LibUtils.tick();
//            decideTime_ += LibUtils.computeTime(s,e);
            return null;
        }
        else {
            node.function = decisionNeo.function;
            node.decision = decisionNeo;
            node.component = decisionComponent;
            node.level = level_;

            assert (node.function.equals(highTrail_.get(currentLine_).t0.children.get(currentChild_).function));

            //System.out.println("NEO decision = " + decisionNeo.function + " @" + level_ + " node ID = " + node.id + " SAT decision= " + decisionSAT + " assume= " + satUtils_.posLit(decisionSAT));

            Pair<Integer,Integer> p = new Pair<Integer,Integer>(currentLine_,currentChild_);
            Pair<Node, Pair<Integer,Integer>> p2 = new Pair<Node, Pair<Integer,Integer>>(node, p);

            trailNeo_.add(p2);
            trailSAT_.push(-decisionSAT);
            assert (satUtils_.getSolver().truthValue(decisionSAT) != Lbool.FALSE);
            if (satUtils_.getSolver().truthValue(decisionSAT) == Lbool.UNDEFINED)
                satUtils_.getSolver().assume(satUtils_.posLit(decisionSAT));

            currentSATLevel_.add(satUtils_.getSolver().decisionLevel());

            for (int i = 0; i < decisionNeo.inputs.length; i++) {
                trail_.get(level_).add(new Pair<Node, Integer>(node.children.get(i), level_));
            }
            level_++;

//            long e = LibUtils.tick();
//            decideTime_ += LibUtils.computeTime(s,e);
            return node;
        }
    }

    public Node decideHigh(){

        //long s = LibUtils.tick();

        assert (level_ < highTrail_.size());
        Production decisionNeo = null;
        int decisionSAT = -1;
        int decisionComponent = -1;

        Node node = highTrail_.get(level_).t0;
        Map<String, Pair<Production, Integer>> decideMap = new HashMap<>();
        List<String> decideDomain = new ArrayList<>();
        for (Production p : node.domain) {
            int var = varNodes_.get(new Pair<Integer, Production>(node.id, p));

            if (satUtils_.getSolver().truthValue(var) == Lbool.UNDEFINED ||
                    satUtils_.getSolver().truthValue(var) == Lbool.TRUE) {
                decideMap.put(p.function, new Pair<Production, Integer>(p, var));
                decideDomain.add(p.function);
            }
        }

        if (!decideDomain.isEmpty()) {
            String decision = nextDecisionHigher(decideDomain);
            if (decision == null){
                level_ = -1;
                return null;
            }
            if (decision == ""){
                // we need to go to the next program
                if (level_ != 0) {
                    backtrackStep2(0, false, false);
                }
                level_ = -2;
                return null;
            }
            Pair<Production, Integer> p = decideMap.get(decision);
            decisionNeo = p.t0;
            decisionSAT = p.t1;
            decisionComponent = p.t0.id;
        }

        if (decisionNeo == null) {
//            long e = LibUtils.tick();
//            decideTime_ += LibUtils.computeTime(s,e);
           return null;
        } else {
            node.function = decisionNeo.function;
            node.decision = decisionNeo;
            node.component = decisionComponent;
            node.level = ++level_;

            //System.out.println("NEO decision = " + decisionNeo.function + " @" +level_ + " node ID = " + node.id + " SAT decision= " + decisionSAT + " assume= " + satUtils_.posLit(decisionSAT));

            Pair<Integer,Integer> p = new Pair<Integer,Integer>(currentLine_,currentChild_);
            Pair<Node, Pair<Integer,Integer>> p2 = new Pair<Node, Pair<Integer,Integer>>(node, p);

            trailNeo_.add(p2);
            trailSAT_.push(-decisionSAT);
            assert (satUtils_.getSolver().truthValue(decisionSAT) != Lbool.FALSE);
            if (satUtils_.getSolver().truthValue(decisionSAT) == Lbool.UNDEFINED)
                satUtils_.getSolver().assume(satUtils_.posLit(decisionSAT));

            currentSATLevel_.add(satUtils_.getSolver().decisionLevel());

            for (int i = 0; i < decisionNeo.inputs.length; i++) {
                trail_.get(level_-1).add(new Pair<Node, Integer>(node.children.get(i), level_));
            }
//            long e = LibUtils.tick();
//            decideTime_ += LibUtils.computeTime(s,e);
            return node;
        }
    }


    private boolean blockModel() {

        boolean unsat = false;

        unsat = backtrackStep2(0, true, true);
        step_ = 1;
        if (unsat)
            System.out.println("s UNSATISFIABLE : backtracking block model");

        //System.out.println("#constraints = " + satUtils_.getSolver().nConstraints());
        return unsat;
    }

    private boolean blockModelNeo() {

        // TODO: ongoing work for backtracking less
        //return blockModel();
        boolean unsat = false;
        unsat = undoLast();
        if (unsat)
            System.out.println("s UNSATISFIABLE : backtracking block model");

        //System.out.println("#constraints = " + satUtils_.getSolver().nConstraints());
        return unsat;


//        boolean unsat = false;
//
//        //unsat = backtrackStep2(level_-1, true, false);
//        unsat = backtrackStep2(0, true, false);
//        step_ = backtrackStep(level_);
//        if (unsat)
//            System.out.println("s UNSATISFIABLE : backtracking block model");
//
//        //System.out.println("#constraints = " + satUtils_.getSolver().nConstraints());
//        return unsat;
    }

    private boolean backtrackStep1(int lvl, boolean block) {

//        long s = LibUtils.tick();

        // There is a disparity between the level in Neo and the level in the SAT solvers
        // Several decisions in Neo may be in the same internal level in the SAT solver
        // When backtracking, we need to take into consideration the internals of the SAT solver
        int backtrack_lvl = lvl;
        // FIXME: potential issue with on the fly adding of clauses
//        while (currentSATLevel_.get(level_) == currentSATLevel_.get(backtrack_lvl) || backtrack_lvl > 0) {
//            backtrack_lvl--;
//        }
        backtrack_lvl = 0;

        assert (trailNeo_.size() > 0 && trailSAT_.size() > 0);
        int size = trailNeo_.size();

        if (backtrack_lvl < highTrail_.size()) {
            for (int i = backtrack_lvl; i < highTrail_.size(); i++)
                trail_.get(i).clear();
        }

        satUtils_.getSolver().cancelUntil(currentSATLevel_.get(backtrack_lvl));

        boolean conflict = false;
        if (block) conflict = satUtils_.blockTrail(trailSAT_);

        for (int i = size; i > backtrack_lvl; i--) {
            // undo
            trailNeo_.get(trailNeo_.size() - 1).t0.function = "";
            trailNeo_.get(trailNeo_.size() - 1).t0.decision = null;
            trailNeo_.get(trailNeo_.size() - 1).t0.level = -1;

            trailNeo_.remove(trailNeo_.size() - 1);
            //trailSAT_.remove(trailSAT_.size() - 1);
            trailSAT_.pop();
        }
        level_ = backtrack_lvl;
        currentSATLevel_.subList(backtrack_lvl+1,currentSATLevel_.size()).clear();
        assert (currentSATLevel_.size() == level_ + 1);

//        long e = LibUtils.tick();
//        backtrackTime1_ += LibUtils.computeTime(s,e);
//        backtrackTime_ += LibUtils.computeTime(s,e);

        return conflict;
    }

    private boolean backtrackStep2(int lvl, boolean block, boolean sat) {

//        long s = LibUtils.tick();

        // There is a disparity between the level in Neo and the level in the SAT solvers
        // Several decisions in Neo may be in the same internal level in the SAT solver
        // When backtracking, we need to take into consideration the internals of the SAT solver


        int backtrack_lvl = lvl;
        if (sat) {
//            while (currentSATLevel_.get(level_) == currentSATLevel_.get(backtrack_lvl) && backtrack_lvl > 0) {
//                backtrack_lvl--;
//            }
            backtrack_lvl = 0;
        }

        assert (trailNeo_.size() > 0 && trailSAT_.size() > 0);
        int size = trailNeo_.size();

//        long ss1 = LibUtils.tick();
        if (backtrack_lvl < highTrail_.size()) {
            for (int i = backtrack_lvl; i < highTrail_.size(); i++)
                trail_.get(i).clear();
        }
//        long ee1 = LibUtils.tick();
//        backtrackTimeTrail_ += LibUtils.computeTime(ss1,ee1);

//        long ss2 = LibUtils.tick();
        satUtils_.getSolver().cancelUntil(currentSATLevel_.get(backtrack_lvl));
//        long ee2 = LibUtils.tick();
//        backtrackTimeSAT_ += LibUtils.computeTime(ss2,ee2);

//        long ss3 = LibUtils.tick();
        boolean conflict = false;
        if (block){
            VecInt cc = new VecInt();
            for (int i = 0; i < trailSAT_.size()-1; i++){
                cc.push(trailSAT_.get(i));
            }

            if (!assignmentsCache_.containsKey(cc.toString())){
                assignmentsCache_.put(cc.toString(),new HashSet());
            }
            assignmentsCache_.get(cc.toString()).add(trailSAT_.get(trailSAT_.size()-1));
            if (step_ != 4)
                conflict = satUtils_.blockTrail(trailSAT_);
        }
//        long ee3 = LibUtils.tick();
//        backtrackTimeBlock_ += LibUtils.computeTime(ss3,ee3);

//        long ss4 = LibUtils.tick();
        for (int i = size; i > backtrack_lvl; i--) {
            // undo
            trailNeo_.get(trailNeo_.size() - 1).t0.function = "";
            trailNeo_.get(trailNeo_.size() - 1).t0.decision = null;
            trailNeo_.get(trailNeo_.size() - 1).t0.level = -1;

            currentLine_ = trailNeo_.get(trailNeo_.size() - 1).t1.t0;
            currentChild_ = trailNeo_.get(trailNeo_.size() - 1).t1.t1;

            trailNeo_.remove(trailNeo_.size() - 1);
            //trailSAT_.remove(trailSAT_.size() - 1);
            trailSAT_.pop();
        }
//        long ee4 = LibUtils.tick();
//        backtrackTimeTrailNeo_ += LibUtils.computeTime(ss4,ee4);

//        long ss5 = LibUtils.tick();
        level_ = backtrack_lvl;
        currentSATLevel_.subList(backtrack_lvl+1,currentSATLevel_.size()).clear();
        assert (currentSATLevel_.size() == level_ + 1);
//        long ee5 = LibUtils.tick();
//        backtrackTimeOther_ += LibUtils.computeTime(ss5,ee5);

//        long e = LibUtils.tick();
//        backtrackTime2_ += LibUtils.computeTime(s,e);
//        backtrackTime_ += LibUtils.computeTime(s,e);

        return conflict;
    }

    private boolean undoLast() {

        boolean conflict = false;

        VecInt cc = new VecInt();
        for (int i = 0; i < trailSAT_.size()-1; i++){
            cc.push(trailSAT_.get(i));
        }

        if (!assignmentsCache_.containsKey(cc.toString())){
            assignmentsCache_.put(cc.toString(),new HashSet());
        }
        assignmentsCache_.get(cc.toString()).add(trailSAT_.get(trailSAT_.size()-1));

        // undo just the last decision?
        currentChild_--;
        if (currentChild_ < 0){
            currentChild_ = 0;
            currentLine_--;
            if (currentLine_ < 0){
                currentLine_ = 0;
                step_ = 2;
                backtrackStep2(highTrail_.size(), false, false);
            } else {
                int lvl = findLevel(currentLine_);
                backtrackStep2(lvl, false, false);
                step_ = backtrackStep(lvl);
                if (step_ != 3){
                    currentLine_ = 0;
                    currentChild_ = 0;
                }
            }
        } else {
            int lvl = findLevel(currentLine_);
            backtrackStep2(lvl, false, false);
            step_ = backtrackStep(lvl);
            if (step_ != 3){
                currentLine_ = 0;
                currentChild_ = 0;
            }
        }
        return conflict;
    }


    private int findLevel(int line){
        int bt = 0;
        for (int i = trailNeo_.size()-1; i >= 0; i--){
            //if (trailNeo_.get(i).t1.t0 == line && trailNeo_.get(i).t1.t1 == 0) {
            if (trailNeo_.get(i).t1.t0 == line) {
                bt = trailNeo_.get(i).t0.level;
                break;
            }
        }
        return bt;
    }

    private Pair<Node, Node> translate(int line) {

//        long s = LibUtils.tick();

        intermediateLearning_.clear();
        treeLearning_ = false;

        Node current = null;
        Object startNode = grammar_.start();
        Node root = new Node();
        root.setSymbol(startNode);
        root.function = "root";
        root.id=0;
        root.component=0;
        root.setConcrete(true);

        partial_ = false;

        List<Node> ast = new ArrayList<>();
        for (int i = 0; i < highTrail_.size(); i++){
            Node node = highTrail_.get(i).t0;
            Node ast_node = new Node();

            assert(node.function.compareTo("")!=0);

            ast_node.id = node.id;
            ast_node.component = node.component;
            ast_node.function = node.function;
            ast_node.setSymbol(prodSymbols_.get(ast_node.function));

            int children = 0;
            for (Node c : node.children) {
                if (!c.function.equals("")) {
                    children++;
                }
            }

            if (prodName_.get(node.function).inputs.length != children) {
                partial_ = true;
            }

            if (prodName_.get(node.function).inputs.length == children) {
                ast_node.setConcrete(true);
            }

            if (i == line){
                current = new Node();
                current.id = ast_node.id;
                current.component = ast_node.component;
                current.function = ast_node.function;
                current.setSymbol(prodSymbols_.get(ast_node.function));
                current.setConcrete(ast_node.isConcrete());
            }

            children = 0;
            for (Node c: node.children){
                if (!c.function.equals("")) {
                    if (c.function.startsWith("line")){
                        String[] parts = c.function.split("line");
                        // Assumes only 1 digit
                        int index = Character.getNumericValue(parts[1].charAt(0));
                        ast_node.addChild(ast.get(index));
                        intermediateLearning_.put(node.id,new Pair<Integer,String>(c.id,c.function));
                    } else {
                        Node ch = new Node();
                        ch.function = c.function;
                        ch.component = c.component;
                        ch.id = c.id;
                        ch.setSymbol(prodSymbols_.get(ch.function));
                        assert (prodName_.containsKey(ch.function));
                        if (prodName_.get(ch.function).inputs.length == c.children.size()){
                            ch.setConcrete(true);
                        }
                        ast_node.addChild(ch);
                    }
                    children++;
                }
            }

            ast.add(ast_node);
        }

        assert(!ast.isEmpty());
        root.addChild(ast.get(ast.size()-1));
        //System.out.println("P' = " + root);
        if (current != null) {
//            System.out.println("current' = " + current.id);
//            System.out.println("current' = " + current.function);
        }

        mapnew2old_.clear();
        mapold2new_.clear();

        List<Node> bfs = new ArrayList<>();
        bfs.add(root);
        while (!bfs.isEmpty()) {
            Node node = bfs.remove(bfs.size() - 1);
            //System.out.println("node = " + node + " id = " + node.id + " function = " + node.function);
            for (int i = 0; i < node.children.size(); i++) {
                Node child = node.children.get(i);
                //System.out.println("child = " + child);
                if (binaryComponent_.contains(child.function))
                    treeLearning_ = true;
                Pair p = new Pair<>(i+1,node.id);
                assert (map2ktree_.containsKey(p));

                mapold2new_.put(child.id,map2ktree_.get(p));
                mapnew2old_.put(map2ktree_.get(p),child.id);
                child.id = map2ktree_.get(p);

                bfs.add(child);
            }
        }

        if (current != null){
            assert (mapold2new_.containsKey(current.id));
            current.id = mapold2new_.get(current.id);
//            System.out.println("current = " + current.id);
//            System.out.println("current = " + current.function);
        }

        //printTree(root);
        //System.out.println("P = " + root);
        assert (root.id == 0);
//        System.out.println("current' = " + current);
        Pair<Node,Node> result = new Pair<Node,Node>(root,current);
//        long e = LibUtils.tick();
//        translateTime_ += LibUtils.computeTime(s,e);

        return result;
    }

    private void printTree(Node root) {
        List<Node> bfs = new ArrayList<>();
        bfs.add(root);
        while (!bfs.isEmpty()) {
            Node node = bfs.remove(bfs.size() - 1);
            //assert (!node.domain.isEmpty());
            System.out.println("Node " + node.id + " function= " + node.function + " concrete= " + node.isConcrete());
            for (Production p : node.domain) {
                System.out.println("Node " + node.id + " | Production= " + p.function);
            }
            for (int i = 0; i < node.children.size(); i++)
                bfs.add(node.children.get(i));
        }
    }

    private void printProgram(Node node){
        String program = node.function;
        program += " ( ";
        for (Node n : node.children){
            if (!n.function.equals("")) {
                program += n.function + " ";
            }
        }
        program += " ) ";
        System.out.println("Line= " + program);
    }

    public int convertLevelFromSATtoNeo(int lvl){
        int neo_lvl = lvl;
        for (int i = 0; i < currentSATLevel_.size(); i++){
            if (currentSATLevel_.get(i) == lvl){
                neo_lvl = i;
                break;
            }
        }
        return neo_lvl;
    }


    public int backtrackStep(int lvl){
        if (lvl < highTrail_.size())
            return 1;
        else if (lvl < step2lvl_)
            return 2;
        else
            return 3;
    }

    public void cacheAST(String program, boolean block){
        assert (!cacheAST_.containsKey(program));
        //if (!cacheAST_.containsKey(program))
        cacheAST_.put(program, block);

    }

    public void printSketches() {

        boolean unsat = false;
        while (!unsat) {
            while (level_ < highTrail_.size()) {

                if (unsat) break;

                //long s = LibUtils.tick();
                Constr conflict = satUtils_.propagate();
                //long e = LibUtils.tick();
                //propagateTime_ += LibUtils.computeTime(s,e);
                if (conflict != null) {
                    int backjumpLevel = satUtils_.analyzeSATConflict(conflict);
                    int neoLevel = convertLevelFromSATtoNeo(backjumpLevel);
                    if (backjumpLevel == -1) {
                        unsat = true;
                        break;
                    } else backtrackStep1(neoLevel, false);
                } else {
                    // No conflict
                    Node decision = decideHigh();
                    if (decision == null) {
                        if (level_ == 0) {
                            unsat = true;
                            break;
                        }

                        while (backtrackStep1(level_ - 1, true)) {
                            if (level_ == 0) {
                                unsat = true;
                                break;
                            }
                        }
                    }

                }
            }

            if (unsat) {
                System.out.println("s NO SOLUTION");
                break;
            }

            step_ = 2;

            String sketch = "";
            for (int i = 0; i < highTrail_.size(); i++) {
                assert (highTrail_.get(i).t0.function != "");
                sketch += highTrail_.get(i).t0.function + " ";
            }
            if (!sketches_.containsKey(sketch)) {
                sketches_.put(sketch, true);
                System.out.println("Sketch #" + sketches_.size() + ": " + sketch);
            }
            unsat = blockModel();
        }
    }


    public Pair<Node,Node> search() {



        Node result = null;
        boolean unsat = false;

//        printSketches();
//        unsat = true;

        while (!unsat) {

            if (step_ == 1) {
                //long s1 = LibUtils.tick();
                // STEP 1. Decide all higher-order components
                while (level_ < highTrail_.size()) {

                    if (unsat) break;

                    //long s = LibUtils.tick();
                    Constr conflict = satUtils_.propagate();
                    //long e = LibUtils.tick();
                    //propagateTime_ += LibUtils.computeTime(s,e);
                    if (conflict != null) {
                        int backjumpLevel = satUtils_.analyzeSATConflict(conflict);
                        int neoLevel = convertLevelFromSATtoNeo(backjumpLevel);
                        if (backjumpLevel == -1) {
                            unsat = true;
                            break;
                        } else backtrackStep1(neoLevel, false);
                    } else {
                        // No conflict
                        Node decision = decideHigh();

                        if (level_ == -2){
                            // FIXME: quick hack to go to next sketch
                            level_ = 0;
                            assert (decision == null);
                            // go to next sketch
                            continue;
                        }

                        if (decision == null && level_ == 0){
                            unsat = true;
                            break;
                        }

                        if (level_ == -1){ // FIXME: quick hack to exit after all programs are checked
                            unsat = true;
                            break;
                        }

                        if (level_ != 0) { // FIXME: quick hack to go to the next program
                            if (decision == null) {
                                while (backtrackStep1(level_ - 1, true)) {
                                    if (level_ == 0) {
                                        unsat = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                if (unsat) {
                    System.out.println("s NO SOLUTION");
                    break;
                }

                step_ = 2;

                String sketch =  "";
                for (int i = 0; i < highTrail_.size(); i++){
                    assert (highTrail_.get(i).t0.function != "");
                    sketch += highTrail_.get(i).t0.function + " ";
                }

                if (!sketches_.containsKey(sketch)){

                    assignmentsCache_.clear();
                    sketches_.put(sketch, true);

                    if (learning_) {
                        List<Integer> next_skt = new ArrayList<>();
                        currentSketch_.clear();
                        for (int i = 0; i < highTrail_.size(); i++) {
                            assert (!highTrail_.get(i).t0.decision.equals(""));
                            int v = varNodes_.get(new Pair<Integer, Production>(highTrail_.get(i).t0.id, highTrail_.get(i).t0.decision));
                            next_skt.add(-v);
                            Pair<Integer, String> pp = new Pair<Integer, String>(highTrail_.get(i).t0.id, highTrail_.get(i).t0.function);
                            currentSketch_.add(pp);
                        }

//                        backtrackStep1(0, false);
//                        step_ = 1;

                        SATUtils.getInstance().cleanLearnts();
                        cacheCore_.clear();
                        //SATUtils.getInstance().cleanVariables();

                        if (SATUtils.getInstance().getSolver().nConstraints() > 600000) {
                            backtrackStep1(0, false);
                            step_ = 1;
                            //SATUtils.getInstance().cleanLearnts();
                            SATUtils.getInstance().cleanLocals();
                            //SATUtils.getInstance().cleanEqLearnts();
                        }

                        if (!currentSketchClause_.isEmpty()) {
                            SATUtils.getInstance().addClause(currentSketchClause_, SATUtils.ClauseType.SKTASSIGNMENT);
                            currentSketchClause_.clear();
                        }
                        for (Integer l : next_skt)
                            currentSketchClause_.push(l);

                        /*
                        boolean conflict = SATUtils.getInstance().addEqLearnts(currentSketch_, sketchNodes_);
                        if (conflict) {
                            unsat = true;
                            System.out.println("s NO SOLUTION");
                            break;
                        }
                        Constr conf = satUtils_.propagate();
                        if (conf != null) {
                            backtrackStep1(0, false);
                            step_ = 1;
                        }
                        */

                    }
                    System.out.println("Sketch #iterations = " + iterations_);
                    iterations_ = 0;
                    System.out.println("Sketch #" + sketches_.size() + ": " + sketch);
                    //Z3Utils.getInstance().cleanCache();
                    System.out.println("#constraints = " + SATUtils.getInstance().getSolver().nConstraints());
                } else {
                    if (iterations_ > ITERATION_LIMIT){
                        // go to next sketch
                        backtrackStep1(0,false);
                        step_ = 1;
                        SATUtils.getInstance().addClause(currentSketchClause_, SATUtils.ClauseType.ASSIGNMENT);
                    }
                }

//                long e1 = LibUtils.tick();
//                step1Time_ += LibUtils.computeTime(s1,e1);
            }

            if (step_ == 2) {

//                long s2 = LibUtils.tick();

                // STEP 2. Decide on all inputs/lines
                currentLine_ = 0;
                currentChild_ = 0;
                boolean repeat_step2 = false;

                while (currentLine_ < trail_.size()) {
                    boolean children_assigned = false;
                    while (currentChild_ < trail_.get(currentLine_).size()) {
//                        System.out.println("currentChild = " + currentChild_ + " currentLine=" + currentLine_);

//                        long s = LibUtils.tick();
                        Constr conflict = satUtils_.propagate();
//                        long e = LibUtils.tick();
//                        propagateTime_ += LibUtils.computeTime(s,e);
                        if (conflict != null) {
                            int backjumpLevel = satUtils_.analyzeSATConflict(conflict);
                            int neoLevel = convertLevelFromSATtoNeo(backjumpLevel);
                            if (backjumpLevel == -1) {
                                unsat = true;
                                break;
                            }

                            if (backjumpLevel < highTrail_.size()) {
                                backtrackStep2(neoLevel, false, false);
                                step_ = 1;
                                break;
                            } else {
                                backtrackStep2(neoLevel, false, false);
                                if (level_ < highTrail_.size()) {
                                    step_ = 1;
                                    break;
                                } else {
                                    repeat_step2 = true;
                                    break;
                                }
                            }

                        } else {
                            // Assumes that all higher-order components have children
                                Node decision = decideInputs();
                                if (decision != null)
                                    children_assigned = true;
                            currentChild_++;
                        }
                    }

                    if (unsat) {
                        System.out.println("s NO SOLUTION");
                        break;
                    }

                    if (!children_assigned && step_ == 2 && !repeat_step2) {
                        // was not possible to assign children
                        // should we go bgo back to step1?
                            step_ = 1;
                            currentLine_ = 0;
                            currentChild_ = 0;
                            backtrackStep2(0, true, true);
                            break;

                    } else {

                        if (step_ == 1 || repeat_step2) {
                            currentChild_ = 0;
                            currentLine_ = 0;
                            break;
                        }
                        currentChild_ = 0;
                        currentLine_++;
                    }
                }

                if (step_ != 1 && !repeat_step2 && step_ == 2) {

                    // Check that we are in a consistent state
//                    long s = LibUtils.tick();
                    Constr conflict = satUtils_.propagate();
//                    long e = LibUtils.tick();
//                    propagateTime_ += LibUtils.computeTime(s,e);

                    if (conflict != null) {
                        int backjumpLevel = satUtils_.analyzeSATConflict(conflict);
                        int neoLevel = convertLevelFromSATtoNeo(backjumpLevel);
                        if (backjumpLevel == -1) {
                            unsat = true;
                            break;
                        } else backtrackStep2(neoLevel, false, false);
                        step_ = backtrackStep(neoLevel);
                        currentLine_ = trailNeo_.get(trailNeo_.size()-1).t1.t0;
                        currentChild_ = trailNeo_.get(trailNeo_.size()-1).t1.t1+1;
                        if (currentChild_ >= trail_.get(currentLine_).size()){
                            currentLine_++;
                            currentChild_=0;
                        }
                   } else {

                        assert (conflict == null);
                        step_ = 3;

                        currentChild_ = 0;
                        currentLine_ = 0;
                        step2lvl_ = level_;

                        Pair<Node,Node> ast = translate(-1);
                        if (!cacheAST_.containsKey(ast.t0.toString())) {
                            ast_ = ast;
                            iterations_++;
//                            System.out.println("Propagate time=:" + (propagateTime_));
//                            System.out.println("Backtrack time=:" + (backtrackTime_));
//                            System.out.println("Learning time=:" + (learnTime_));
//                            System.out.println("Translate time=:" + (translateTime_));
//                            System.out.println("Decide time=:" + (decideTime_));
//                            System.out.println("Step1 time=:" + (step1Time_));
//                            System.out.println("Step2 time=:" + (step2Time_));
//                            System.out.println("Step3 time=:" + (step3Time_));
//
//                            System.out.println("backtrackTime2_ time=:" + (backtrackTime2_));
//                            System.out.println("backtrackTime1_ time=:" + (backtrackTime1_));
//                            System.out.println("backtrackTimeOther_ time=:" + (backtrackTimeOther_));
//                            System.out.println("backtrackTimeTrailNeo_ time=:" + (backtrackTimeTrailNeo_));
//                            System.out.println("backtrackTimeBlock_ time=:" + (backtrackTimeBlock_));
//                            System.out.println("backtrackTimeSAT_ time=:" + (backtrackTimeSAT_));
//                            System.out.println("backtrackTimeTrail_ time=:" + (backtrackTimeTrail_));
//
                            cpTrailSAT_.clear();
                            trailSAT_.copyTo(cpTrailSAT_);

                            if (!treeSketch_.equals("")){
                                if (!ast.toString().equals(treeSketch_)){
                                    Z3Utils.getInstance().cleanCache();
                                }
                            }
                            treeSketch_ = ast.toString();

                            return ast;
                        }

                    }
                }

//                long e2 = LibUtils.tick();
//                step2Time_ += LibUtils.computeTime(s2,e2);
            }


            if (step_ == 3) {

//                long s3 = LibUtils.tick();

                // Fill line-by-line and only ask the deduction system after we have a full line
                    //assert (currentLine_ < trail_.size());
                    while (currentChild_ < trail_.get(currentLine_).size()) {
//                        long s = LibUtils.tick();
                        Constr conflict = satUtils_.propagate();
//                        long e = LibUtils.tick();
//                        propagateTime_ += LibUtils.computeTime(s,e);
                        if (conflict != null) {
                            int backjumpLevel = satUtils_.analyzeSATConflict(conflict);
                            int neoLevel = convertLevelFromSATtoNeo(backjumpLevel);
                            if (backjumpLevel == -1) {
                                unsat = true;
                                break;
                            } else backtrackStep2(neoLevel, false, false);
                            step_ = backtrackStep(neoLevel);

                            currentLine_ = trailNeo_.get(trailNeo_.size()-1).t1.t0;
                            currentChild_ = trailNeo_.get(trailNeo_.size()-1).t1.t1+1;
                            assert(currentLine_ < trail_.size());
                            if (currentChild_ >= trail_.get(currentLine_).size()){
                                currentLine_++;
                                currentChild_=0;
                            }

                            if (currentLine_ == trail_.size()){
                                // go to step 2?
                                step_ = 2;
                                currentLine_ = 0;
                                currentChild_ = 0;
                                backtrackStep2(highTrail_.size(), true, true);
                                step_ = backtrackStep(level_);
                                break;
                            }

                            if (step_ != 3){
                                break;
                            }
                        } else {
                            if (highTrail_.get(currentLine_).t0.children.get(currentChild_).function.equals("")) {
                                Node decision = decideFirst();
                                if (decision == null) {
                                    // go back to step 2?
                                    step_ = 2;
                                    currentLine_ = 0;
                                    currentChild_ = 0;
                                    backtrackStep2(highTrail_.size(), true, true);
                                    step_ = backtrackStep(level_);
                                    break;
                                } else {
                                    assert (!highTrail_.get(currentLine_).t0.children.get(currentChild_).function.equals(""));
                                }
                                currentChild_++;
                            } else {
                                currentChild_++;
                            }
                        }
                    }

                if (step_ == 3) {

                    if (unsat) {
                        System.out.println("s NO SOLUTION");
                        break;
                    }

                    // Check that we are in a consistent state
                    //long s = LibUtils.tick();
                    Constr conflict = satUtils_.propagate();
//                    long e = LibUtils.tick();
//                    propagateTime_ += LibUtils.computeTime(s,e);
                    if (conflict != null) {
                        int backjumpLevel = satUtils_.analyzeSATConflict(conflict);
                        int neoLevel = convertLevelFromSATtoNeo(backjumpLevel);
                        if (backjumpLevel == -1) {
                            unsat = true;
                            break;
                        } else backtrackStep2(neoLevel, false, false);
                        step_ = backtrackStep(neoLevel);

                        if (step_ == 3){
                            // go back one line
                            //assert(currentLine_ > 0);
                            if (currentLine_ == 0){
                                // go back to step 2
                                step_ = 2;
                                currentLine_ = 0;
                                currentChild_ = 0;
                                backtrackStep2(highTrail_.size(), false, false);
                            } else {
                                currentLine_--;
                                currentChild_ = 0;
                                int lvl = findLevel(currentLine_);
                                backtrackStep2(lvl, false, false);
                                step_ = backtrackStep(lvl);
                                if (step_ != 3){
                                    currentLine_ = 0;
                                    currentChild_ = 0;
                               }
                            }
                        }
                    } else {

                        Pair<Node,Node> ast = translate(currentLine_);
                        step_ = 4; // Line is complete

                        if (cacheAST_.containsKey(ast.t0.toString())) {
                            assert (partial_);
                            step_ = 3;
                        } else {
                            if (learning_) {
                                if (currentLine_ < learntLine_) {
                                    satUtils_.getInstance().cleanLearnts(currentLine_);
                                    blockLearnFlag_ = true;
                                    clauseLearn_.clear();
                                    for (Pair<Integer, Integer> p : blockLearn_) {
                                        if (p.t1 <= currentLine_)
                                            clauseLearn_.push(-p.t0);
                                    }
                                    learntAst_ = blockAst_;
                                }

                                learntLine_ = currentLine_;

                                blockLearn_.clear();
                                for (int i = 0; i < highTrail_.size(); i++) {
                                    Node node = highTrail_.get(i).t0;
                                    if (!node.function.equals("")) {
                                        int v = varNodes_.get(new Pair<Integer, Production>(node.id, node.decision));
                                        blockLearn_.add(new Pair<Integer, Integer>(v, 0));
                                    }

                                    for (Node n : node.children) {
                                        if (!n.function.equals("")) {
                                            if (n.function.startsWith("input") || n.function.startsWith("line")) {
                                                int v = varNodes_.get(new Pair<Integer, Production>(n.id, n.decision));
                                                blockLearn_.add(new Pair<Integer, Integer>(v, 0));
                                            } else {
                                                int v = varNodes_.get(new Pair<Integer, Production>(n.id, n.decision));
                                                blockLearn_.add(new Pair<Integer, Integer>(v, i));
                                            }
                                        }
                                    }
                                }
                            }


                            blockAst_ = ast.t0;
                            ast_ = ast;
                            iterations_++;
//                            System.out.println("Propagate time=:" + (propagateTime_));
//                            System.out.println("Backtrack time=:" + (backtrackTime_));
//                            System.out.println("Learning time=:" + (learnTime_));
//                            System.out.println("Translate time=:" + (translateTime_));
//                            System.out.println("Decide time=:" + (decideTime_));
//                            System.out.println("Step1 time=:" + (step1Time_));
//                            System.out.println("Step2 time=:" + (step2Time_));
//                            System.out.println("Step3 time=:" + (step3Time_));
//
//                            System.out.println("backtrackTime2_ time=:" + (backtrackTime2_));
//                            System.out.println("backtrackTime1_ time=:" + (backtrackTime1_));
//                            System.out.println("backtrackTimeOther_ time=:" + (backtrackTimeOther_));
//                            System.out.println("backtrackTimeTrailNeo_ time=:" + (backtrackTimeTrailNeo_));
//                            System.out.println("backtrackTimeBlock_ time=:" + (backtrackTimeBlock_));
//                            System.out.println("backtrackTimeSAT_ time=:" + (backtrackTimeSAT_));
//                            System.out.println("backtrackTimeTrail_ time=:" + (backtrackTimeTrail_));

                            cpTrailSAT_.clear();
                            trailSAT_.copyTo(cpTrailSAT_);
                            return ast;
                        }

                        // Go to the next line of code
                        currentLine_++;
                        currentChild_ = 0;

                    }
                }

//                long e3 = LibUtils.tick();
//                step3Time_ += LibUtils.computeTime(s3,e3);
            }

            }

            return null;
        }


    }
