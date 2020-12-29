package org.genesys.decide;

import com.microsoft.z3.BoolExpr;
import org.genesys.language.Grammar;
import org.genesys.language.Production;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.models.Trio;
import org.genesys.utils.SATUtils;
import org.sat4j.core.VecInt;
import org.sat4j.minisat.core.*;
import org.sat4j.specs.Lbool;
import org.sat4j.specs.TimeoutException;

import java.lang.reflect.Array;
import java.util.*;

/**
 * Created by ruben on 7/6/17.
 */
public class NeoSolver implements AbstractSolver<BoolExpr, Node> {

    private Decider decider_;

    private SATUtils satUtils_;

    private Grammar grammar_;

    private int maxLen_ = 4;

    private int nodeId_ = 1;

    private int learnts_ = 0;

    private boolean init_ = false;

    /* Maps types to productions */
    private Map<String, List<Production>> prodTypes_ = new HashMap<>();

    /* Maps production to symbols */
    private Map<String, Object> prodSymbols_ = new HashMap<>();

    /* Stores ancestor information */
    private Map<Node, List<Node>> ancestors_ = new HashMap<>();

    /* Stores parent information */
    private Map<Node, Node> parents_ = new HashMap<>();

    /* Order of exploration */
    private List<Integer> exploration_ids_ = new ArrayList<>();
    private List<Node> exploration_nodes_ = new ArrayList<>();

    /* Domain of productions */
    private final List<Production> domainProductions_ = new ArrayList<>();
    private final List<Production> domainLeafProductions_ = new ArrayList<>();
    private final List<Production> domainRootProductions_ = new ArrayList<>();

    /* Maps variables to node id */
    private final Map<Pair<Integer, Production>, Integer> varNodes_ = new HashMap<>();
    private final Map<Pair<Integer, String>, Integer> coreNodes_ = new HashMap<>();

    /* String to production */
    private Map<String, Production> prodName_ = new HashMap<>();

    /* Nodes */
    private final List<Node> leafNodes_ = new ArrayList<>();
    private final List<Node> nodes_ = new ArrayList<>();
    private Node root_ = null;

    /* Producions from inputs */
    private final List<Production> inputProductions_ = new ArrayList<>();

    /* Trail */
    //private final Deque<Pair<Node, Integer>> trail_ = new LinkedList<>();
    private final List<Pair<Node, Integer>> trail_ = new ArrayList<>();
    private final List<Node> trailNeo_ = new ArrayList<>();
    private final List<Integer> trailSAT_ = new ArrayList<>();

    private int maxChildren_ = 0;

    private int nbProductions_ = 0;

    private int nbVariables_ = 0;

    private int currentLevel_ = 0;

    private boolean partial_ = false;

    private Node ast_ = null;

    private List<Integer> currentSATLevel_ = new ArrayList<>();

    public NeoSolver(Grammar g, Decider decider) {
        satUtils_ = SATUtils.getInstance();
        grammar_ = g;
        decider_ = decider;
        Object start = grammar_.start();
    }

    public NeoSolver(Grammar g, int depth, Decider decider) {
        satUtils_ = SATUtils.getInstance();
        maxLen_ = depth;
        grammar_ = g;
        decider_ = decider;
        Object start = grammar_.start();
    }

    @Override
    public void cacheAST(String program, boolean block) {};

    @Override
    public ArrayList<Double> getLearnStats() {return null; };

    public boolean learnCore(List<Pair<Integer, List<String>>> core) {
        boolean conflict = false;

        HashMap<Integer,String> node2function = new HashMap<>();
        List<Node> bfs = new ArrayList<>();
        Node root = ast_;
        bfs.add(root);
        while (!bfs.isEmpty()) {
            Node node = bfs.remove(bfs.size() - 1);
            node2function.put(node.id,node.function);
            for (int i = 0; i < node.children.size(); i++)
                bfs.add(node.children.get(i));
        }

        List<List<Integer>> eqClauses = new ArrayList<>();
        String learnt = "";
        for (Pair<Integer,List<String>> p : core){
            List<Integer> eq = new ArrayList<>();
            learnt = learnt + "[(" + p.t0 + ") ";
            //System.out.println("node= " + p.t0);
            assert (node2function.containsKey(p.t0));
            learnt = learnt + node2function.get(p.t0);
            Pair<Integer, String> id = new Pair<>(p.t0,node2function.get(p.t0));
            assert (coreNodes_.containsKey(id));
            eq.add(coreNodes_.get(id));
            //System.out.println("function= " + node2function.get(p.t0));
            // FIXME : this should not hapen!
            if (!node2function.get(p.t0).contains("input")) {
                for (String l : p.t1) {
                    Pair<Integer, String> id2 = new Pair<>(p.t0, l);
                    if (!coreNodes_.containsKey(id2))
                        continue;
                    //assert (coreNodes_.containsKey(id2));
                    eq.add(coreNodes_.get(id2));
                    learnt = learnt + " , " + l;
                }
            }
            eqClauses.add(eq);
            learnt = learnt + "]";
        }
        if (!eqClauses.isEmpty()) {
            System.out.println("Learning: " + learnt);
                conflict = SATUtils.getInstance().learnCoreGlobal(eqClauses);
        }
        return conflict;

    }

    @Override
    public Node getModel(BoolExpr core, boolean block) {

        if (!init_) {
            init_ = true;
            loadGrammar();
            initDataStructures();
        } else {
            if (block) {
                boolean conflict = blockModel();
                if (conflict)
                    return null;
            }
            partial_ = false;
        }

        Node node = search();
        return node;
    }

    @Override
    public Node getCoreModel(List<Pair<Integer, List<String>>> core, boolean block, boolean global) {

        if (!init_) {
            init_ = true;
            loadGrammar();
            initDataStructures();
        } else {
            boolean conflict = blockModel();
            if (conflict)
                return null;
            else {
                boolean confl = learnCore(core);
                if (confl){
                    System.out.println("s UNSATISFIABLE - learning core");
                    return null;
                }
            }
            partial_ = false;
        }

        Node node = search();
        return node;
    }

    @Override
    public Node getCoreModelSet(List<List<Pair<Integer, List<String>>>> core, boolean block, boolean global) {
        return null;
    }


    public String nextDecision(List<String> ancestors, List<String> domain) {

        assert (domain.size() > 1);
//        System.out.println("Ancestors = " + ancestors.toString());
//        System.out.println("Domain = " + domain.toString());

        // Choose your favorite statistical heuristic here!

        String decision = "";
        for (String d : domain){
            if (d.contains("input")) {
                decision = d;
                break;
            }
        }

        if (decision.equals("")) {
            decision = decider_.decide(ancestors, domain);
        }
        //String decision = domain.get(0);

//        System.out.println("Decision = " + decision);

        assert (!decision.equals(""));
        return decision;

    }

    private boolean blockModel() {

        //System.out.println("trail_.size = " + trailNeo_.size());
        boolean unsat = false;
//        while (backtrack(currentLevel_ - 1)) {
//            if (currentLevel_ == 0) {
//                System.out.println("s UNSATISFIABLE : backtracking block model");
//                unsat = true;
//                break;
//            }
//        }
        unsat = backtrack(0, true);
        if (unsat)
            System.out.println("s UNSATISFIABLE : backtracking block model");
        assert (trailNeo_.size() == currentLevel_);
        return unsat;
    }

    private void initDataStructures() {
        // build the k-tree
        root_ = createNode(domainRootProductions_, true);
        root_.id = 0;

        // depth starts at 2 since root is at level 1
        createTree(root_, 2);
        computeAncestors();

        //explorationStrategy(root_);

        // create Boolean variables
        createVariables(root_);

        //printTree(root_);

        // Create empty SAT solver
        satUtils_.createSolver();
        satUtils_.createVars(nbVariables_);

        buildGrammarSATFormula();

        // Decision level for Neo and SAT solver
        currentLevel_ = 0;
        currentSATLevel_.add(0);

        // Nodes that still need to be processed
        trail_.add(new Pair<Node, Integer>(root_, currentLevel_));

    }

    private Node createNode(List<Production> productions, boolean children) {
        Node node = new Node("", new ArrayList<>(), productions);
        if (children) {
            for (int i = 0; i < maxChildren_; i++) {
                Node child = new Node("", new ArrayList<>(), productions);
                node.addChild(child);
            }
        }
        return node;
    }

    /* Assumes that the root was already created */
    private void createTree(Node node, int depth) {
        assert (!nodes_.contains(node));
        nodes_.add(node);
        node.id = nodeId_++;

        assert (node.children.size() == maxChildren_);
        for (int i = 0; i < node.children.size(); i++) {
            if (depth == maxLen_) {
                // leaf node
                node.children.get(i).setDomain(domainLeafProductions_);
                leafNodes_.add(node);
                assert (((Node) node.children.get(i)).children.isEmpty());
            } else {
                node.children.get(i).setDomain(domainProductions_);
                assert (((Node) node.children.get(i)).children.isEmpty());
                for (int j = 0; j < maxChildren_; j++)
                    ((Node) node.children.get(i)).addChild(createNode(domainProductions_, false));
            }
        }

        if (depth < maxLen_) {
            for (int i = 0; i < node.children.size(); i++)
                createTree(((Node) node.children.get(i)), depth + 1);
        } else {
            for (int i = 0; i < node.children.size(); i++) {
                node.children.get(i).id = nodeId_++;
                assert (!nodes_.contains(node.children.get(i)));
                nodes_.add(node.children.get(i));
            }
        }
    }

    private <T> void createVariables(Node node) {

        for (Production p : node.domain) {
            Pair<Integer, Production> pair = new Pair<Integer, Production>(node.id, p);
            Pair<Integer, String> pair2 = new Pair<Integer, String>(node.id, p.function);
            varNodes_.put(pair, ++nbVariables_);
            coreNodes_.put(pair2,nbVariables_);
        }

        for (int i = 0; i < node.children.size(); i++)
            createVariables(node.children.get(i));
    }

    private void printTree(Node root) {
        List<Node> bfs = new ArrayList<>();
        bfs.add(root);
        while (!bfs.isEmpty()) {
            Node node = bfs.remove(bfs.size() - 1);
            //assert (!node.domain.isEmpty());
            System.out.println("Node " + node.id + " concrete= " + node.isConcrete());
            for (Production p : node.domain) {
                System.out.println("Node " + node.id + " | Production= " + p.function);
            }
            for (int i = 0; i < node.children.size(); i++)
                bfs.add(node.children.get(i));
        }
    }

    public boolean isPartial(){
        return partial_;
    }

    private boolean backtrack(int lvl, boolean block) {

        // There is a disparity between the level in Neo and the level in the SAT solvers
        // Several decisions in Neo may be in the same internal level in the SAT solver
        // When backtracking, we need to take into consideration the internals of the SAT solver
        int backtrack_lvl = lvl;
        while (currentSATLevel_.get(currentLevel_) == currentSATLevel_.get(backtrack_lvl) || backtrack_lvl > 0) {
            backtrack_lvl--;
        }

        assert (trailNeo_.size() > 0 && trailSAT_.size() > 0);
        int size = trailNeo_.size();

        for (Iterator<Pair<Node, Integer>> iter = trail_.iterator(); iter.hasNext(); ) {
            Pair<Node, Integer> p = iter.next();
            if (p.t1 >= backtrack_lvl)
                iter.remove();
        }
        if (backtrack_lvl == 0)
            trail_.add(new Pair<Node, Integer>(root_,0));

        satUtils_.getSolver().cancelUntil(currentSATLevel_.get(backtrack_lvl));

        boolean conflict = false;

        VecInt clause = new VecInt();
        for (int i = 0; i < trailSAT_.size(); i++) {
            clause.push(-trailSAT_.get(i));
        }

        if (block) conflict = satUtils_.blockTrail(clause);

        for (int i = size; i > backtrack_lvl; i--) {
            // undo
            trailNeo_.get(trailNeo_.size() - 1).function = "";
            trailNeo_.get(trailNeo_.size() - 1).decision = null;
            trailNeo_.get(trailNeo_.size() - 1).level = -1;

            trailNeo_.remove(trailNeo_.size() - 1);
            trailSAT_.remove(trailSAT_.size() - 1);
        }
        currentLevel_ = backtrack_lvl;
        currentSATLevel_.subList(backtrack_lvl+1,currentSATLevel_.size()).clear();
        assert (currentSATLevel_.size() == currentLevel_ + 1);

        return conflict;
    }

    private boolean orphanParent() {

        boolean orphan = false;
        for (Node node : trailNeo_) {
            for (int i = 0; i < node.decision.inputs.length; i++) {
                if (node.children.get(i).function.compareTo("") == 0) {
                    trail_.add(new Pair<Node, Integer>(node.children.get(i), currentLevel_));
                    orphan = true;
                }

            }
        }
        return orphan;
    }

    private boolean inputsUsed() {

        boolean used = false;
        for (Production p : inputProductions_) {
            ArrayList<Integer> clause = new ArrayList<>();
            for (Node node : nodes_) {
                if (node.function.contains("input")){
                    used = true;
                    break;
                }

            }
            if (used)
                break;
        }
        return used;

//        boolean used = true;
//        for (Production p : inputProductions_) {
//            ArrayList<Integer> clause = new ArrayList<>();
//            for (Node node : nodes_) {
//                Pair<Integer, Production> pair = new Pair<Integer, Production>(node.id, p);
//                if (varNodes_.containsKey(pair)) {
//                    clause.add(varNodes_.get(pair));
//                }
//            }
//            boolean ok = false;
//            for (Integer var : clause) {
//                if (satUtils_.getSolver().truthValue(var) == Lbool.TRUE) {
//                    ok = true;
//                    break;
//                }
//            }
//            used = used && ok;
//        }

//          return used;
    }

    private int naiveAnalyzeSATConflict(Constr conflict) {

        int backjumpLevel = currentLevel_;

        if (currentLevel_ == 0) {
            // unsat
            backjumpLevel = -1;
        } else {
            while (backtrack(currentLevel_ - 1, true)) {
                if (currentLevel_ == 0) {
                    System.out.println("s UNSATISFIABLE : naive analyze SAT conflict");
                    backjumpLevel = -1;
                    break;
                }
            }
        }

        if (backjumpLevel != -1)
            backjumpLevel = currentLevel_;
        return backjumpLevel;
    }


    public Node search() {

        Node result = null;

        boolean unsat = false;
        while (!unsat) {
            while (!trail_.isEmpty()) {

                if (partial_) break;

                if (unsat) break;

                Constr conflict = propagate();

                if (conflict != null) {
                    int backjumpLevel = satUtils_.analyzeSATConflict(conflict);
                    if (backjumpLevel == -1){
                        System.out.println("s UNSATISFIABLE : SAT Conflict");
                        unsat = true;
                        break;
                    } else backtrack(backjumpLevel, false);
                } else {
                    // No conflict
                    Node decision = decide(trail_);
                    if (decision == null) {
                        if (currentLevel_ == 0) {
                            System.out.println("s UNSATISFIABLE : lvl = 0");
                            unsat = true;
                            break;
                        }

                        while (backtrack(currentLevel_ - 1,true)) {
                            if (currentLevel_ == 0) {
                                System.out.println("s UNSATISFIABLE : backtracking");
                                unsat = true;
                                break;
                            }
                        }
                    }
                    if (inputsUsed()){
                        partial_ = true;
                    }
                }
            }

            if (partial_) {
                if (!orphanParent())
                    partial_ = false;
                break;
            }

            if (unsat) break;

            if (!orphanParent()) {

                if (inputsUsed()) {
                    //System.out.println("s SATISFIABLE");
                    break;
                } else {
                //System.out.println("INPUTS NOT USED");
                    Constr conflict = propagate();
                    if (conflict != null) {
                            //System.out.println("SAT Conflict");
                            int backjumpLevel = satUtils_.analyzeSATConflict(conflict);
                            //int backjumpLevel = naiveAnalyzeSATConflict(conflict);
                            if (backjumpLevel == -1){
                                System.out.println("s UNSATISFIABLE : SAT Conflict");
                                unsat = true;
                                break;
                            } else backtrack(backjumpLevel, false);
                    } else {
                        while (backtrack(currentLevel_ - 1,true)) {
                            if (currentLevel_ == 0) {
                                System.out.println("s UNSATISFIABLE : backtracking inputs");
                                unsat = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (!unsat) {
            result = translate(root_);
        }

        ast_ = result;
        return result;
    }

    private void computeAncestors() {

        for (Node node : nodes_)
            ancestors_.put(node, new ArrayList<>());

        LinkedList<Node> worklist = new LinkedList<>();
        worklist.add(root_);
        while (!worklist.isEmpty()) {
            Node node = worklist.pollFirst();
            for (Node child : node.children) {
                List<Node> list = ancestors_.get(child);
                for (Node previous : ancestors_.get(node))
                    list.add(previous);
                list.add(node);
                ancestors_.put(child, list);
                worklist.add(child);
                parents_.put(child, node);
            }
        }

    }

    private Node translate(Node node) {

        LinkedList<Pair<Node, Node>> worklist = new LinkedList<>();
        Object startNode = grammar_.start();
        Node root = new Node();
        root.setSymbol(startNode);
        root.function = "root";
        root.id=0;
        root.setConcrete(true);
        Node child = new Node();
        child.function = node.function;
        child.setSymbol(prodSymbols_.get(child.function));
        child.id=node.id;
        root.addChild(child);


        int children = 0;
        for (Node c : node.children) {
            if (!c.function.equals("")) {
                children++;
                worklist.add(new Pair<Node, Node>(c, child));
            }
        }

        if (prodName_.get(node.function).inputs.length == children)
            child.setConcrete(true);

        while (!worklist.isEmpty()) {
            Pair<Node, Node> p = worklist.pollFirst();
            if (p.t0.function.compareTo("") != 0) {
                Node ch = new Node();
                ch.function = p.t0.function;
                ch.id=p.t0.id;
                ch.setSymbol(prodSymbols_.get(ch.function));
                p.t1.addChild(ch);
                children = 0;
                for (Node c : p.t0.children) {
                    if (!c.function.equals("")) {
                        worklist.add(new Pair<Node, Node>(c, ch));
                        children++;
                    }
                }
                if (prodName_.get(ch.function).inputs.length == children)
                    ch.setConcrete(true);
            }
        }

        //printTree(root);

        return root;
    }

    private List<String> getAncestors(Node node) {

        ArrayList<String> ancestors = new ArrayList<>();
        for (Node n : ancestors_.get(node)) {
            ancestors.add(n.function);
            assert (n.function.compareTo("") != 0);
        }
        return ancestors;

    }


    private Node decide(List<Pair<Node, Integer>> trail) {

        assert (!trail.isEmpty());
        Node node = decideNode(trail);

        Production decisionNeo = null;
        int decisionSAT = -1;

        Map<String, Pair<Production, Integer>> decideMap = new HashMap<>();
        List<String> decideDomain = new ArrayList<>();
        for (Production p : node.domain) {
            int var = varNodes_.get(new Pair<Integer, Production>(node.id, p));

            if (satUtils_.getSolver().truthValue(var) == Lbool.UNDEFINED ||
                    satUtils_.getSolver().truthValue(var) == Lbool.TRUE) {
                //System.out.println("var= " + var + " value=" + satUtils_.getSolver().truthValue(var));
                decideMap.put(p.function, new Pair<Production, Integer>(p, var));
                decideDomain.add(p.function);
            }
        }

//        for (int i = 1; i <= satUtils_.getNbVars(); i++){
//            System.out.println("v= " + i + " value= " + satUtils_.getSolver().truthValue(i));
//        }


//        System.out.println("level= " + currentLevel_);
//        for (Node n : trailNeo_) {
//            System.out.println(n.function);
//        }
//        for (String s : decideDomain) {
//            System.out.println("Domain= " + s);
//        }

        if (decideDomain.size() == 1) {
            Pair<Production, Integer> p = decideMap.get(decideDomain.get(0));
            decisionNeo = p.t0;
            decisionSAT = p.t1;
        } else if (decideDomain.size() > 1) {
            String decision = nextDecision(getAncestors(node), decideDomain);
            Pair<Production, Integer> p = decideMap.get(decision);
            decisionNeo = p.t0;
            decisionSAT = p.t1;
        }

        if (decisionNeo == null)
            return null;
        else {

            node.function = decisionNeo.function;
            node.decision = decisionNeo;
            node.level = ++currentLevel_;
            //System.out.println("NEO decision = " + decisionNeo.function + " @" + currentLevel_ + " node ID = " + node.id + " SAT decision= " + decisionSAT + " assume= " + satUtils_.posLit(decisionSAT));

            trailNeo_.add(node);
            trailSAT_.add(decisionSAT);
            assert (satUtils_.getSolver().truthValue(decisionSAT) != Lbool.FALSE);
            if (satUtils_.getSolver().truthValue(decisionSAT) == Lbool.UNDEFINED)
                satUtils_.getSolver().assume(satUtils_.posLit(decisionSAT));

            currentSATLevel_.add(satUtils_.getSolver().decisionLevel());

            for (int i = 0; i < decisionNeo.inputs.length; i++) {
                trail.add(new Pair<Node, Integer>(node.children.get(i), currentLevel_));
            }

            // check if siblings are either in the trail or assigned
            // FIXME: this is not efficient
            if (node != root_) {
                Node parent = parents_.get(node);
//                System.out.println("Parent = " + parent.function);
//                System.out.println("children = " + parent.decision.inputs.length);
                for (int i = 0; i < parent.decision.inputs.length; i++) {

                    //System.out.println("child = " + i + " has " + parent.children.get(i).function);
                    if (parent.children.get(i) == node)
                        continue;

                    boolean intrail = false;
                    for (Pair<Node, Integer> p : trail) {
                        if (p.t0 == parent.children.get(i)) {
                            intrail = true;
                            break;
                        }
                    }

                    boolean assigned = false;
                    if (!intrail) {
                        if (parent.children.get(i).function.compareTo("") != 0) {
                            assigned = true;
                        }
                    }
                    //System.out.println("assigned = " + assigned + " intrail= " + intrail);
                    if (!assigned && !intrail) {
                        trail.add(new Pair<Node, Integer>(parent.children.get(i), currentLevel_));
                    }
                }
            }

            return node;
        }
    }

    private Node decideNode(List<Pair<Node, Integer>> pending) {

        int id = pending.get(0).t0.id;
        int min = id;
        int pos = 0;
        //System.out.println("head = " + id);
        for (int i = 1; i < pending.size(); i++){
            if (pending.get(i).t0.id < min) {
                min = id;
                pos = i;
            }
        }
        //return pending.pollFirst().t0;

        Node n  = pending.remove(pos).t0;
        return n;


    }


    private Constr propagate() {

        Constr satConflict = satUtils_.getSolver().propagate();
        return satConflict;
    }

    private void buildGrammarSATFormula() {

//        // AMO production at each node is used
//        for (Node node : nodes_) {
//            VecInt clause = new VecInt();
//            for (Production p : node.domain) {
//                Pair<Integer, Production> pair = new Pair<Integer, Production>(node.id, p);
//                assert (varNodes_.containsKey(pair));
//                clause.push(varNodes_.get(pair));
//            }
//            satUtils_.addAMO(clause);
//        }

        // If a production is used in a parent node then this implies restrictions on the children
        for (Node node : nodes_) {
            for (Production p : node.domain) {

                int productionVar = varNodes_.get(new Pair<Integer, Production>(node.id, p));

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
                    if (clause.size() > 1)
                        satUtils_.addClause(clause);

                    for (Production pc : node.children.get(i).domain) {
                        if (!occurs.contains(pc)) {
                            VecInt lits = new VecInt(new int[]{-productionVar, -varNodes_.get(new Pair<Integer, Production>(node.children.get(i).id, pc))});
                            // Parent restricts the domain of child (negatively)
                            satUtils_.addClause(lits);
                        }
                    }
                }

                // If this node contains less than k children then the remaining will be empty
                if (!node.children.isEmpty()) {
                    for (int i = p.inputs.length; i < maxChildren_; i++) {
                        for (Production pc : node.children.get(i).domain) {
                            VecInt lits = new VecInt(new int[]{-productionVar, -varNodes_.get(new Pair<Integer, Production>(node.children.get(i).id, pc))});
                            satUtils_.addClause(lits);
                        }
                    }
                }
            }
        }

        // Inputs must be used in some node
//        for (Production p : inputProductions_) {
//            VecInt clause = new VecInt();
//            for (Node node : nodes_) {
//                Pair<Integer, Production> pair = new Pair<Integer, Production>(node.id, p);
//                if (varNodes_.containsKey(pair)) {
//                    clause.push(varNodes_.get(pair));
//                }
//            }
//            satUtils_.addClause(clause);
//        }

        Constr conflict = satUtils_.propagate();
        assert (conflict == null);
    }

    private <T> void loadGrammar() {
        List<Production<T>> prods = grammar_.getProductions();
        prods.addAll(grammar_.getInputProductions());

        inputProductions_.addAll((List<Production<T>>) grammar_.getInputProductions());

        for (Production<T> prod : prods) {
            nbProductions_++;

            if (!prodTypes_.containsKey(prod.source.toString())) {
                prodTypes_.put(prod.source.toString(), new ArrayList<Production>());
            }

            prodTypes_.get(prod.source.toString()).add(prod);

            prodSymbols_.put(prod.function, prod.source);

            prodName_.put(prod.function, prod);

            if (prod.inputs.length > maxChildren_)
                maxChildren_ = prod.inputs.length;

            domainProductions_.add(prod);
            if (prod.inputs.length == 0)
                domainLeafProductions_.add(prod);

            if (grammar_.getOutputType().toString().compareTo(prod.source.toString()) == 0)
                domainRootProductions_.add(prod);
        }
    }

    private void explorationStrategy(Node node){

        Deque<Node> queue = new LinkedList<>();
        queue.add(node);

        while (!queue.isEmpty()){
            Node n = queue.poll();
            exploration_ids_.add(n.id);
            System.out.println("id = " + n.id);

            // DFS using the left node -- is the higher-order component always on the left?
            for (int i = n.children.size()-1; i >= 0; i--) {
                queue.addFirst(n.children.get(i));
            }
        }



    }

}
