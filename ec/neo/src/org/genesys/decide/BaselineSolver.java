package org.genesys.decide;

import com.microsoft.z3.BoolExpr;
import com.microsoft.z3.Expr;
import com.microsoft.z3.FuncDecl;
import com.microsoft.z3.Model;
import org.genesys.language.Grammar;
import org.genesys.language.Production;
import org.genesys.models.MultivalueMap;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.models.Trio;
import org.genesys.utils.LibUtils;
import org.genesys.utils.Z3Utils;

import java.util.*;

/**
 * Created by yufeng on 5/31/17.
 */
public class BaselineSolver implements AbstractSolver<BoolExpr, Node> {

    private Z3Utils z3Utils;

    private Grammar grammar_;

    private int maxLen_ = 4;

    /* Control variables for productions */
    private final Map<String, Production> prodCtrlMap = new HashMap<>();

    private final MultivalueMap<Production, String> prodCtrlInvMap = new MultivalueMap<>();

    private final MultivalueMap<String, BoolExpr> inputMap = new MultivalueMap<>();

    /* Control variables for symbols */
    private final Map<BoolExpr, String> symCtrlMap = new HashMap<>();

    private Model model_;

    public BaselineSolver(Grammar g) {
        z3Utils = Z3Utils.getInstance();
        grammar_ = g;
        Object start = grammar_.start();
//        System.out.println("start symbol:" + start);
        Trio<Integer, BoolExpr, BoolExpr> formula = generate(grammar_, start, maxLen_);
        BoolExpr formulaConjoin = z3Utils.conjoin(formula.t1, z3Utils.getVarById("bool_0"));
        //all input vars must be used.
        for (String inputKey : inputMap.keySet()) {
            BoolExpr inputCst = z3Utils.disjoin(LibUtils.listToArray(inputMap.get(inputKey)));
            formulaConjoin = z3Utils.conjoin(formulaConjoin, inputCst);
        }
//        System.out.println("Big formula: " + formulaConjoin);
        z3Utils.init(formulaConjoin);
    }

    public BaselineSolver(Grammar g, int depth) {
        maxLen_ = depth;
        z3Utils = Z3Utils.getInstance();
        grammar_ = g;
        Object start = grammar_.start();
        Trio<Integer, BoolExpr, BoolExpr> formula = generate(grammar_, start, maxLen_);
        BoolExpr formulaConjoin = z3Utils.conjoin(formula.t1, z3Utils.getVarById("bool_0"));
        //all input vars must be used.
        for (String inputKey : inputMap.keySet()) {
            BoolExpr inputCst = z3Utils.disjoin(LibUtils.listToArray(inputMap.get(inputKey)));
            formulaConjoin = z3Utils.conjoin(formulaConjoin, inputCst);
        }
        z3Utils.init(formulaConjoin);
    }

    @Override
    public void cacheAST(String program, boolean block) {};

    @Override
    public ArrayList<Double> getLearnStats() {return null; };

    @Override
    public boolean isPartial() { return false; }

    @Override
    public Node getCoreModel(List<Pair<Integer, List<String>>> core, boolean block, boolean global) {
        return null;
    }

    @Override
    public Node getCoreModelSet(List<List<Pair<Integer, List<String>>>> core, boolean block, boolean global) {
        return null;
    }


        @Override
    public Node getModel(BoolExpr core, boolean block) {
        if (model_ != null) {
            boolean hasNext = z3Utils.blockSolution();
            if (!hasNext) return null;
        }
        model_ = z3Utils.getModel();
//        System.out.println("Current model:" + model_);
        Node ast = translate(model_);
        return ast;
    }

    private <T> Trio<Integer, BoolExpr, BoolExpr> generate(Grammar grammar, T s, int len) {
        List<Production<T>> prods = grammar.productionsFor(s);
//        System.out.println(s + "--->" + prods);
        List<BoolExpr> exactList = new ArrayList<>();
        List<BoolExpr> conjoinList = new ArrayList<>();
        /* Control variable for current symbol. */
        BoolExpr parentVar = z3Utils.getFreshBoolVar();
        symCtrlMap.put(parentVar, s.toString());
        if (len == 0) {
            if (prods.isEmpty())//terminal can be the leaf.
                return new Trio<>(len, z3Utils.trueExpr(), parentVar);
            else //non-terminal cant be the leaf
                return new Trio<>(len, z3Utils.neg(parentVar), parentVar);
        }
        len--;

        for (Production<T> prod : prods) {
            BoolExpr prodVar = z3Utils.getFreshBoolVar();
            prodCtrlMap.put(prodVar.toString(), prod);
            prodCtrlInvMap.add(prod, prodVar.toString());
            //Track input constraints.
            String prodFunc = prod.function;
            if (prodFunc != null && prodFunc.contains("input")) {
                inputMap.add(prodFunc, prodVar);
            }
            //System.out.println(prodVar + " mapsto%%%%%%%: " + prod);
            /* create a fresh var for each production. */
            for (T child : prod.inputs) {
                Trio<Integer, BoolExpr, BoolExpr> subResult = generate(grammar, child, len);
                BoolExpr childSymVar = subResult.t2;
                symCtrlMap.put(childSymVar, child.toString());
                /* AND edge: if production 'prod' happen, it's equivalent to say all its children symbols also happen. */
                conjoinList.add(z3Utils.imply(prodVar, childSymVar));
                conjoinList.add(z3Utils.imply(childSymVar, prodVar));

                /* Conjoin children's constraints. */
                BoolExpr subExpr = subResult.t1;
                conjoinList.add(subExpr);
            }
            exactList.add(prodVar);
        }

        /* Only one production can happen. */
        BoolExpr[] exactArray = LibUtils.listToArray(exactList);

        /* conjoin current constraints with the children. */
        BoolExpr[] conjoinArray = LibUtils.listToArray(conjoinList);

        BoolExpr childToParent = z3Utils.imply(z3Utils.disjoin(exactArray), parentVar);

        BoolExpr exactExpr = z3Utils.exactOne(exactArray);
        /* Current symbol implies that exact one production can happen */
        exactExpr = z3Utils.imply(parentVar, exactExpr);
        BoolExpr conjoinExpr = z3Utils.conjoin(conjoinArray);

        if (exactArray.length > 0)
            conjoinExpr = z3Utils.conjoin(conjoinExpr, childToParent);

        Trio<Integer, BoolExpr, BoolExpr> result = new Trio<>(len, z3Utils.conjoin(exactExpr, conjoinExpr), parentVar);
        return result;
    }

    /**
     * Translate Z3 model into a concrete AST node.
     *
     * @param m: Z3 model.
     * @return AST node
     */
    private Node translate(Model m) {
        if (m == null) return null;

//        System.out.println("begin to translate: " + m);
        LinkedList<String> models = new LinkedList<>();
        for (FuncDecl fd : m.getConstDecls()) {
            Expr val = m.getConstInterp(fd);
            if (val.equals(z3Utils.trueExpr())) {
                String prodId = fd.getName().toString();
                if (prodCtrlMap.containsKey(prodId))
                    models.add(fd.getName().toString());
            }
        }

        Node sol = extractAst(models);
        return sol;
    }

    private Node extractAst(LinkedList<String> models) {
        /* Important invariant: need to traverse the control variable s in ascending order! */
        Collections.sort(models, (lhs, rhs) -> {
            assert lhs.contains("_") && rhs.contains("_");
            /* compare the id of two control variables. */
            int lhsIdx = Integer.parseInt(lhs.split("_")[1]);
            int rhsIdx = Integer.parseInt(rhs.split("_")[1]);
            return (lhsIdx - rhsIdx);
        });

        LinkedList<Pair<Object, Node>> worklist = new LinkedList<>();
        int count = 1;
        Object startNode = grammar_.start();
        Node root = new Node();
        root.id = count;
        count++;
        root.setSymbol(startNode);
        Pair<Object, Node> rootPair = new Pair<>(startNode, root);
        worklist.add(rootPair);
        while (!worklist.isEmpty()) {
            Pair<Object, Node> workerPair = worklist.pollFirst();
            Object workerSym = workerPair.t0;
            Node workerNode = workerPair.t1;

            for (Object o : grammar_.productionsFor(workerSym)) {

                Production prod = (Production) o;
                assert prodCtrlInvMap.containsKey(prod) : prod;
                List<String> ctrlKeys = prodCtrlInvMap.get(prod);

                if (!models.isEmpty() && ctrlKeys.contains(models.getFirst())) {
                    models.removeFirst();
                    workerNode.function = prod.function;

                    List<Pair<Object, Node>> childList = new ArrayList();
                    for (Object childSym : prod.inputs) {
                        Node childNode = new Node();
                        childNode.id = count;
                        childNode.setSymbol(childSym);
                        workerNode.addChild(childNode);
                        childList.add(new Pair<>(childSym, childNode));
                        count++;
                    }
                    /* FIXME: Perform DFS without recursion. */
                    if (!childList.isEmpty()) {
                        Collections.reverse(childList);
                        for (Pair<Object, Node> elem : childList) {
                            worklist.addFirst(elem);
                        }
                    }
                    break;
                }
            }
        }

        assert models.isEmpty() : models;
//        System.out.println("Current AST:" + root + " models:" + models);
        return root;
    }

}
