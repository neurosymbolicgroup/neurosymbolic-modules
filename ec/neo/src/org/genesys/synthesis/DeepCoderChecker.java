package org.genesys.synthesis;

import com.google.gson.Gson;
import com.microsoft.z3.BoolExpr;
import krangl.DataFrame;
import krangl.GroupedDataFrame;
import org.apache.commons.lang3.StringUtils;
import org.genesys.interpreter.DeepCoderInterpreter;
import org.genesys.interpreter.Interpreter;
import org.genesys.models.*;
import org.genesys.type.Maybe;
import org.genesys.utils.LibUtils;
import org.genesys.utils.MorpheusUtil;
import org.genesys.utils.Z3Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;

/**
 * Created by yufeng on 6/3/17.
 */
public class DeepCoderChecker implements Checker<Problem, List<Pair<Integer, List<String>>>> {

    private HashMap<String, Component> components_ = new HashMap<>();

    private Gson gson = new Gson();

    private Z3Utils z3_ = Z3Utils.getInstance();

    private MorpheusUtil util_ = MorpheusUtil.getInstance();

    private Interpreter interpreter_ = new DeepCoderInterpreter();

    private Map<Pair<Integer, String>, List<BoolExpr>> cstCache_ = new HashMap<>();

    private final String alignId_ = "alignOutput";

    //Properties: LEN, MAX, MIN, FIRST, LAST
    private String[] spec = {
            "OUT_LEN_SPEC", "OUT_MAX_SPEC", "OUT_MIN_SPEC", "OUT_FIRST_SPEC", "OUT_LAST_SPEC",
            "IN0_LEN_SPEC", "IN0_MAX_SPEC", "IN0_MIN_SPEC", "IN0_FIRST_SPEC", "IN0_LAST_SPEC",
            "IN1_LEN_SPEC", "IN1_MAX_SPEC", "IN1_MIN_SPEC", "IN1_FIRST_SPEC", "IN1_LAST_SPEC"
    };

    private Map<String, Object> clauseToNodeMap_ = new HashMap<>();
    //Map a clause to its original spec
    private Map<String, String> clauseToSpecMap_ = new HashMap<>();

    public DeepCoderChecker(String specLoc) throws FileNotFoundException {
        File[] files = new File(specLoc).listFiles();
        for (File file : files) {
            assert file.isFile() : file;
            String json = file.getAbsolutePath();
            Component comp = gson.fromJson(new FileReader(json), Component.class);
            components_.put(comp.getName(), comp);
        }
    }

    /**
     * @param specification: Input-output specs.
     * @param node:          Partial AST from Ruben.
     * @return
     */
    @Override
    public boolean check(Problem specification, Node node) {
        return true;
    }

    @Override
    public boolean check(Problem specification, Node node, Node curr) {

        Example example = specification.getExamples().get(0);
        Object output = example.getOutput();
        List inputs = example.getInput();

        /* Generate SMT formula for current AST node. */
        Queue<Node> queue = new LinkedList<>();
        List<BoolExpr> cstList = new ArrayList<>();
        //FIXME: The analysis is not working well.
//        Node rootNode = node.children.get(0);
//        BitSet bit = getCompBits(rootNode);
//        if (bit.isEmpty()) return true;
        queue.add(node);
        while (!queue.isEmpty()) {
            Node worker = queue.remove();
            //Generate constraint between worker and its children.
            String func = worker.function;
            //Get component spec.
            Component comp = components_.get(func);
//            System.out.println("working on : " + func + " id:" + worker.id + " isconcrete:" + worker.isConcrete());
            if ("root".equals(func)) {
                List<BoolExpr> abs = abstractDeepCode(worker, output);
                List<BoolExpr> align = alignOutput(worker);
                cstList.addAll(abs);
                cstList.addAll(align);
            } else if (func.contains("input")) {
                //attach inputs
                List<String> nums = LibUtils.extractNums(func);
                assert !nums.isEmpty();
                int index = Integer.valueOf(nums.get(0));
                Object inDf = inputs.get(index);
                z3_.updateTypeMap(worker.id, worker.function);

                List<BoolExpr> abs = abstractDeepCode(worker, inDf);

                cstList.addAll(abs);
            } else {
                if (!worker.children.isEmpty() && comp != null) {

                    if ((curr != null) && (worker.id == curr.id)) {
                        Maybe<Object> tgt = interpreter_.execute(worker, inputs);
                        if (!tgt.has()) {
                            z3_.clearConflict();
                            return false;
                        }
                        Object obj = tgt.get();
                        if (obj instanceof List) {
                            List objList = (List) obj;
                            if (objList.isEmpty()) {
                                //TODO: type-inhabitant for filter and count
                                z3_.clearConflict();
                                return false;
                            }
                        }
                        List<BoolExpr> abs = abstractDeepCode(worker, tgt.get());
                        if (abs.isEmpty()) {
                            z3_.clearConflict();
                            return false;
                        }
//                        System.out.println("working on PE:" + worker + " res:" + tgt.get());

                        cstList.addAll(abs);
                    }

                    List<BoolExpr> nodeCst = genNodeSpec(worker, comp);
                    cstList.addAll(nodeCst);
                }
            }

            for (int i = 0; i < worker.children.size(); i++) {
                Node child = worker.children.get(i);
                queue.add(child);
            }
        }

        boolean sat = z3_.isSat(cstList, clauseToNodeMap_, clauseToSpecMap_, components_.values());
//        if (!sat) {
//            System.out.println("Prune program:" + node);
//        }
        return sat;
    }

    private List<BoolExpr> genNodeSpec(Node worker, Component comp) {
//        System.out.println("current workder: " + worker.id + " " + worker);
        Pair<Integer, String> key = new Pair<>(worker.id, comp.getName());
        z3_.updateTypeMap(worker.id, comp.getType());
        if (cstCache_.containsKey(key))
            return cstCache_.get(key);
        String[] dest = new String[15];
        String lenVar = "V_LEN" + worker.id;
        String maxVar = "V_MAX" + worker.id;
        String minVar = "V_MIN" + worker.id;
        String firstVar = "V_FIRST" + worker.id;
        String lastVar = "V_LAST" + worker.id;
        dest[0] = lenVar;
        dest[1] = maxVar;
        dest[2] = minVar;
        dest[3] = firstVar;
        dest[4] = lastVar;
        Node child0 = worker.children.get(0);
        String lenChild0Var = "V_LEN" + child0.id;
        String maxChild0Var = "V_MAX" + child0.id;
        String minChild0Var = "V_MIN" + child0.id;
        String firstChild0Var = "V_FIRST" + child0.id;
        String lastChild0Var = "V_LAST" + child0.id;
        dest[5] = lenChild0Var;
        dest[6] = maxChild0Var;
        dest[7] = minChild0Var;
        dest[8] = firstChild0Var;
        dest[9] = lastChild0Var;

        String lenChild1Var = "#";
        String maxChild1Var = "#";
        String minChild1Var = "#";
        String firstChild1Var = "#";
        String lastChild1Var = "#";
        if (worker.children.size() > 1) {
            Node child1 = worker.children.get(1);
            lenChild1Var = "V_LEN" + child1.id;
            maxChild1Var = "V_MAX" + child1.id;
            minChild1Var = "V_MIN" + child1.id;
            firstChild1Var = "V_FIRST" + child1.id;
            lastChild1Var = "V_LAST" + child1.id;
        }
        dest[10] = lenChild1Var;
        dest[11] = maxChild1Var;
        dest[12] = minChild1Var;
        dest[13] = firstChild1Var;
        dest[14] = lastChild1Var;
        List<BoolExpr> cstList = new ArrayList<>();

        for (String cstStr : comp.getConstraint()) {
            String targetCst = StringUtils.replaceEach(cstStr, spec, dest);
            if (targetCst.contains("#")) continue;
            BoolExpr expr = z3_.convertStrToExpr(targetCst);
            cstList.add(expr);
            clauseToNodeMap_.put(expr.toString(), worker.id);
            clauseToSpecMap_.put(expr.toString(), cstStr);
        }
        //cache current cst.
        cstCache_.put(key, cstList);
        return cstList;
    }

    private List<BoolExpr> abstractDeepCode(Node worker, Object obj) {
        List<BoolExpr> cstList = new ArrayList<>();
        String strVal = worker.function;
        if (!"root".equals(worker.function))
            strVal = worker.toString();
        Pair<Integer, String> key = new Pair<>(worker.id, strVal);
        if (cstCache_.containsKey(key)) {
            if (!"root".equals(worker.function) && !worker.function.contains("input")) {
                //Need to also update current assignment.
                List<Pair<Integer, List<String>>> currAssigns = getCurrentAssignment(worker);
                for (BoolExpr o : cstCache_.get(key)) {
                    clauseToNodeMap_.put(o.toString(), currAssigns);
                }
            } else {
                for (BoolExpr o : cstCache_.get(key)) {
                    clauseToNodeMap_.put(o.toString(), worker.id);
                }
            }
            return cstCache_.get(key);
        }

        int len = util_.getLen(obj);
        int max = util_.getMax(obj);
        int min = util_.getMin(obj);
        int first = util_.getFirst(obj);
        int last = util_.getLast(obj);

        String lenVar = "V_LEN" + worker.id;
        String maxVar = "V_MAX" + worker.id;
        String minVar = "V_MIN" + worker.id;
        String firstVar = "V_FIRST" + worker.id;
        String lastVar = "V_LAST" + worker.id;

        BoolExpr lenCst = z3_.genEqCst(lenVar, len);
        BoolExpr maxCst = z3_.genEqCst(maxVar, max);
        BoolExpr minCst = z3_.genEqCst(minVar, min);
        BoolExpr firstCst = z3_.genEqCst(firstVar, first);
        BoolExpr lastCst = z3_.genEqCst(lastVar, last);

        cstList.add(lenCst);
        cstList.add(maxCst);
        cstList.add(minCst);
        cstList.add(firstCst);
        cstList.add(lastCst);

        if ("root".equals(worker.function) || worker.function.contains("input")) {
            clauseToNodeMap_.put(lenCst.toString(), worker.id);
            clauseToNodeMap_.put(maxCst.toString(), worker.id);
            clauseToNodeMap_.put(minCst.toString(), worker.id);
            clauseToNodeMap_.put(firstCst.toString(), worker.id);
            clauseToNodeMap_.put(lastCst.toString(), worker.id);
        } else {
            List<Pair<Integer, List<String>>> currAssigns = getCurrentAssignment(worker);
            clauseToNodeMap_.put(lenCst.toString(), currAssigns);
            clauseToNodeMap_.put(maxCst.toString(), currAssigns);
            clauseToNodeMap_.put(minCst.toString(), currAssigns);
            clauseToNodeMap_.put(firstCst.toString(), currAssigns);
            clauseToNodeMap_.put(lastCst.toString(), currAssigns);
            Set<String> peCore = new HashSet<>();

            peCore.add(lenCst.toString());
            peCore.add(maxCst.toString());
            peCore.add(minCst.toString());
            peCore.add(firstCst.toString());
            peCore.add(lastCst.toString());
            if (MorpheusSynthesizer.learning_ && z3_.hasCache(peCore)) return new ArrayList<>();
        }
        //cache current cst.
        cstCache_.put(key, cstList);
        return cstList;
    }

    private List<BoolExpr> alignOutput(Node worker) {
        Pair<Integer, String> key = new Pair<>(worker.id, alignId_);
        if (cstCache_.containsKey(key))
            return cstCache_.get(key);
        List<BoolExpr> cstList = new ArrayList<>();
        String lenVar = "V_LEN" + worker.id;
        String maxVar = "V_MAX" + worker.id;
        String minVar = "V_MIN" + worker.id;
        String firstVar = "V_FIRST" + worker.id;
        String lastVar = "V_LAST" + worker.id;

        assert worker.children.size() == 1;
        Node lastChild = worker.children.get(0);
        String lenChild0Var = "V_LEN" + lastChild.id;
        String maxChild0Var = "V_MAX" + lastChild.id;
        String minChild0Var = "V_MIN" + lastChild.id;
        String firstChild0Var = "V_FIRST" + lastChild.id;
        String lastChild0Var = "V_LAST" + lastChild.id;

        BoolExpr eqLenCst = z3_.genEqCst(lenVar, lenChild0Var);
        BoolExpr eqMaxCst = z3_.genEqCst(maxVar, maxChild0Var);
        BoolExpr eqMinCst = z3_.genEqCst(minVar, minChild0Var);
        BoolExpr eqFirstCst = z3_.genEqCst(firstVar, firstChild0Var);
        BoolExpr eqLastCst = z3_.genEqCst(lastVar, lastChild0Var);

        cstList.add(eqLenCst);
        cstList.add(eqMaxCst);
        cstList.add(eqMinCst);
        cstList.add(eqFirstCst);
        cstList.add(eqLastCst);
        cstCache_.put(key, cstList);
        return cstList;
    }

    @Override
    public List<Pair<Integer, List<String>>> learnCore() {
        return new ArrayList<>();
    }

    // Given an AST, generate clause for its current assignments
    private List<Pair<Integer, List<String>>> getCurrentAssignment(Node node) {
        List<Pair<Integer, List<String>>> clauses = new ArrayList<>();
        Pair<Integer, List<String>> worker = new Pair<>(node.id, Arrays.asList(node.function));
        clauses.add(worker);
        for (Node child : node.children) {
            clauses.addAll(getCurrentAssignment(child));
        }
        return clauses;
    }

    //0: len; 1: first; 2: last; 3: max; 4: min
    private BitSet getCompBits(Node ast) {
//        System.out.println("checking ast bit:" + ast);
        String compName = ast.function;
        Component component = components_.get(compName);
        String compBit = component.getBit();
        if (compName.startsWith("MAP")) {
            compBit = "10000";
        }
        BitSet compBitSet = LibUtils.fromBitString(compBit);
        BitSet mask = LibUtils.fromBitString("00000");

        if (ast.children.isEmpty()) return compBitSet;

        for (Node child : ast.children) {
            String childName = child.function;
            BitSet childBitSet = LibUtils.fromBitString("11111");
            if (components_.containsKey(childName)) {
                childBitSet = getCompBits(child);
                childBitSet.and(compBitSet);
            } else {
                childBitSet.and(compBitSet);
            }
            mask.or(childBitSet);
        }

        return (BitSet) mask.clone();
    }
}
