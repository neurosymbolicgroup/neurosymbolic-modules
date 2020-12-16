package org.genesys.synthesis;

import com.google.gson.Gson;
import com.microsoft.z3.BoolExpr;
import krangl.DataFrame;
import krangl.GroupedDataFrame;
import org.apache.commons.lang3.StringUtils;
import org.genesys.interpreter.MorpheusValidator2;
import org.genesys.language.MorpheusGrammar;
import org.genesys.models.*;
import org.genesys.utils.LibUtils;
import org.genesys.utils.MorpheusUtil;
import org.genesys.utils.Z3Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;

/**
 * Created by yufeng on 9/3/17.
 * Deduction for Morpheus
 */
public class MorpheusChecker implements Checker<Problem, List<List<Pair<Integer, List<String>>>>> {

    private HashMap<String, Component> components_ = new HashMap<>();

    private Gson gson = new Gson();

    private MorpheusValidator2 validator_;

    private List<List<Pair<Integer, List<String>>>> core_ = new ArrayList<>();

    private Map<String, Object> clauseToNodeMap_ = new HashMap<>();
    //Map a clause to its original spec
    private Map<String, String> clauseToSpecMap_ = new HashMap<>();

    private Z3Utils z3_ = Z3Utils.getInstance();
    private MorpheusUtil util_ = MorpheusUtil.getInstance();

    private Map<Pair<Integer, String>, List<BoolExpr>> cstCache_ = new HashMap<>();

    private final String alignId_ = "alignOutput";

    private String[] spec1 = {"CO_SPEC", "RO_SPEC", "CI0_SPEC", "RI0_SPEC", "CI1_SPEC", "RI1_SPEC"};
    private String[] spec2 = {
            "CO_SPEC", "RO_SPEC", "OG_SPEC", "OH_SPEC", "OC_SPEC",
            "CI0_SPEC", "RI0_SPEC", "IG0_SPEC", "IH0_SPEC", "IC0_SPEC",
            "CI1_SPEC", "RI1_SPEC", "IG1_SPEC", "IH1_SPEC", "IC1_SPEC", "GI0_SPEC"
    };

    public MorpheusChecker(String specLoc, MorpheusGrammar g) throws FileNotFoundException {
        File[] files = new File(specLoc).listFiles();
        for (File file : files) {
            assert file.isFile() : file;
            String json = file.getAbsolutePath();
            Component comp = gson.fromJson(new FileReader(json), Component.class);
            components_.put(comp.getName(), comp);
        }
        validator_ = new MorpheusValidator2(g.getInitProductions());
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
        core_.clear();
        z3_.clearEqClassesInPE();
        Example example = specification.getExamples().get(0);
        Object output = example.getOutput();
        assert output instanceof DataFrame;
        DataFrame outDf = (DataFrame) output;
        List<DataFrame> inputs = example.getInput();

        // Perform type-checking and PE.
        validator_.cleanPEMap();
//        System.out.println("Verifying.... " + node + " curr:" + curr);
        /* Generate SMT formula for current AST node. */
        Queue<Node> queue = new LinkedList<>();
        List<BoolExpr> cstList = new ArrayList<>();

        int currId = 0;
        queue.add(node);
        while (!queue.isEmpty()) {
            Node worker = queue.remove();
            //Generate constraint between worker and its children.
            String func = worker.function;

            //Get component spec.
            Component comp = components_.get(func);
//            System.out.println("working on : " + func + " id:" + worker.id + " isconcrete:" + worker.isConcrete());
            if ("root".equals(func)) {
                List<BoolExpr> abs = abstractTable(worker, outDf, inputs);
                List<BoolExpr> align = alignOutput(worker);
                cstList.addAll(abs);
                cstList.addAll(align);
            } else if (func.contains("input")) {
                //attach inputs
                List<String> nums = LibUtils.extractNums(func);
                assert !nums.isEmpty();
                int index = Integer.valueOf(nums.get(0));
                DataFrame inDf = inputs.get(index);
                z3_.updateTypeMap(worker.id, worker.function);

                List<BoolExpr> abs = abstractTable(worker, inDf, inputs);
                cstList.addAll(abs);
            } else {
                if (!worker.children.isEmpty() && comp != null) {
                    if ((curr != null) && (worker.id == curr.id)) {
                        long start2 = LibUtils.tick();
                        if (worker.children.size() > 1)
                            currId = worker.children.get(1).id;
//                        System.out.println("type on node: " + worker);
                        Pair<Object, List<Map<Integer, List<String>>>> validRes = null;
                        try {
                            validRes = validator_.validate(worker, example.getInput());
                        } catch (Exception e) {
                            System.out.println("ERROR from the R interpreter!!!");
                            return false;
                        }
                        Object judge = validRes.t0;
                        long end2 = LibUtils.tick();
                        MorpheusSynthesizer.typeinhabit += LibUtils.computeTime(start2, end2);
                        if (judge == null) {
                            parseCore(validRes.t1);
//                            System.out.println("prune by type inhabitation: " + worker);
                            return false;
                        } else {
                            DataFrame workerDf = (DataFrame) validRes.t0;
                            if (isValid(workerDf)) {
                                parseCore((validRes.t1));
                                return false;
                            }
                            List<BoolExpr> abs = abstractTable(worker, workerDf, inputs);
                            if (abs.isEmpty()) {
                                z3_.clearConflict();
                                return false;
                            }
                            cstList.addAll(abs);
                        }
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

    @Override
    public List<List<Pair<Integer, List<String>>>> learnCore() {
        return core_;
    }

    private void parseCore(List<Map<Integer, List<String>>> coreList) {
        for (Map<Integer, List<String>> conflict : coreList) {
            List<Pair<Integer, List<String>>> c = new ArrayList<>();
            for (int key : conflict.keySet()) {
                c.add(new Pair<>(key, conflict.get(key)));
            }
            core_.add(c);
        }
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

    private List<BoolExpr> genNodeSpec(Node worker, Component comp) {
//        System.out.println("current workder: " + worker.id + " " + worker);
        Pair<Integer, String> key = new Pair<>(worker.id, comp.getName());
        z3_.updateTypeMap(worker.id, comp.getType());
        if (cstCache_.containsKey(key))
            return cstCache_.get(key);

        String[] dest = new String[16];
        String colVar = "V_COL" + worker.id;
        String rowVar = "V_ROW" + worker.id;
        String groupVar = "V_GROUP" + worker.id;
        String headVar = "V_HEAD" + worker.id;
        String contentVar = "V_CONTENT" + worker.id;
        dest[0] = colVar;
        dest[1] = rowVar;
        dest[2] = groupVar;
        dest[3] = headVar;
        dest[4] = contentVar;
        Node child0 = worker.children.get(0);
        String colChild0Var = "V_COL" + child0.id;
        String rowChild0Var = "V_ROW" + child0.id;
        String groupChild0Var = "V_GROUP" + child0.id;
        String headChild0Var = "V_HEAD" + child0.id;
        String contentChild0Var = "V_CONTENT" + child0.id;
        dest[5] = colChild0Var;
        dest[6] = rowChild0Var;
        dest[7] = groupChild0Var;
        dest[8] = headChild0Var;
        dest[9] = contentChild0Var;

        String colChild1Var = "#";
        String rowChild1Var = "#";
        String groupChild1Var = "#";
        String headChild1Var = "#";
        String contentChild1Var = "#";
        if (worker.children.size() > 1) {
            Node child1 = worker.children.get(1);
            colChild1Var = "V_COL" + child1.id;
            rowChild1Var = "V_ROW" + child1.id;
            groupChild1Var = "V_GROUP" + child1.id;
            headChild1Var = "V_HEAD" + child1.id;
            contentChild1Var = "V_CONTENT" + child1.id;
        }
        dest[10] = colChild1Var;
        dest[11] = rowChild1Var;
        dest[12] = groupChild1Var;
        dest[13] = headChild1Var;
        dest[14] = contentChild1Var;
        dest[15] = "V_ON" + child0.id;

        List<BoolExpr> cstList = new ArrayList<>();

        for (String cstStr : comp.getConstraint()) {
            String targetCst = StringUtils.replaceEach(cstStr, spec2, dest);
            assert !targetCst.contains("#") : targetCst;
            BoolExpr expr = Z3Utils.getInstance().convertStrToExpr(targetCst);
            cstList.add(expr);
            clauseToNodeMap_.put(expr.toString(), worker.id);
            clauseToSpecMap_.put(expr.toString(), cstStr);
        }
        //cache current cst.
        cstCache_.put(key, cstList);
        return cstList;
    }

    private List<BoolExpr> abstractTable(Node worker, DataFrame df, List<DataFrame> inputs) {
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
                if (worker.function.contains("input")) {
                    for (BoolExpr o : cstCache_.get(key)) {
                        clauseToNodeMap_.put(o.toString(), worker.id);
                    }
                }
            }
            return cstCache_.get(key);
        }
        int row = df.getNrow();
        int col = df.getNcol();

        int groupNum = 1;
        int headNum;
        int contentNum;

        if (worker.function.contains("input")) {
            util_.setInputs(inputs);
            headNum = 0;
            contentNum = 0;
        } else {
            if (df instanceof GroupedDataFrame) {
                groupNum = ((GroupedDataFrame) df).getGroups$krangl_main().size();
            }

//            System.out.println("Buggy=============" + worker.function);
            Set head = util_.getHeader(df);
            Set content = util_.getContent(df);
//            System.out.println("content:" + content);
            Set headDiff = new HashSet(head);
            Set contentDiff = new HashSet(content);

            for (DataFrame input : inputs) {
                Set headIn = util_.getHeader(input);
                Set contentIn = util_.getContent(input);
//                System.out.println("input content:" + contentIn);
                contentDiff.removeAll(contentIn);
                headDiff.removeAll(headIn);
                headDiff.removeAll(contentIn);
            }
            headNum = headDiff.size();
            contentNum = contentDiff.size();
//            System.out.println("diffContent:" + contentDiff);
//            System.out.println("worker:" + worker);
//            System.out.println(headNum);
//            System.out.println(groupNum);
//            System.out.println(contentNum);
        }

        String rowVar = "V_ROW" + worker.id;
        String colVar = "V_COL" + worker.id;
        String groupVar = "V_GROUP" + worker.id;
        String headVar = "V_HEAD" + worker.id;
        String contentVar = "V_CONTENT" + worker.id;
        BoolExpr rowCst = z3_.genEqCst(rowVar, row);
        BoolExpr colCst = z3_.genEqCst(colVar, col);

        BoolExpr groupCst = z3_.genEqCst(groupVar, groupNum);
        BoolExpr headCst = z3_.genEqCst(headVar, headNum);
        BoolExpr contentCst = z3_.genEqCst(contentVar, contentNum);

        cstList.add(rowCst);
        cstList.add(colCst);
        //We dont know the group number of the final table.
        if (!"root".equals(worker.function))
            cstList.add(groupCst);

        cstList.add(headCst);
        cstList.add(contentCst);

        BoolExpr onCst = z3_.trueExpr();
        String onVar = "V_ON" + worker.id;
        if (worker.function.equals("group_by")) {
            assert df instanceof GroupedDataFrame;
            GroupedDataFrame gdf = (GroupedDataFrame) df;
            int on = gdf.getBy().size();
            onCst = z3_.genEqCst(onVar, on);
            cstList.add(onCst);
        }

        if ("root".equals(worker.function) || worker.function.contains("input")) {
            clauseToNodeMap_.put(rowCst.toString(), worker.id);
            clauseToNodeMap_.put(colCst.toString(), worker.id);
            clauseToNodeMap_.put(groupCst.toString(), worker.id);
            clauseToNodeMap_.put(headCst.toString(), worker.id);
            clauseToNodeMap_.put(contentCst.toString(), worker.id);
        } else {
            List<Pair<Integer, List<String>>> currAssigns = getCurrentAssignment(worker);
            clauseToNodeMap_.put(rowCst.toString(), currAssigns);
            clauseToNodeMap_.put(colCst.toString(), currAssigns);
            clauseToNodeMap_.put(groupCst.toString(), currAssigns);
            clauseToNodeMap_.put(headCst.toString(), currAssigns);
            clauseToNodeMap_.put(contentCst.toString(), currAssigns);
            Set<String> peCore = new HashSet<>();
            peCore.add(rowCst.toString());
            peCore.add(colCst.toString());
            peCore.add(groupCst.toString());
            peCore.add(headCst.toString());
            peCore.add(contentCst.toString());
            if (worker.function.equals("group_by")) {
                clauseToNodeMap_.put(onCst.toString(), currAssigns);
                peCore.add(onCst.toString());
            }
            if (MorpheusSynthesizer.learning_ && z3_.hasCache(peCore)) return new ArrayList<>();
        }
        //cache current cst.
        cstCache_.put(key, cstList);
        return cstList;
    }

    private boolean isValid(DataFrame df) {
        if (df instanceof GroupedDataFrame) {
            GroupedDataFrame gdf = (GroupedDataFrame) df;
            return gdf.getGroups$krangl_main().size() == 0;
        }
        return (df.getNcol() == 0 || df.getNrow() == 0);
    }

    private List<BoolExpr> alignOutput(Node worker) {
        Pair<Integer, String> key = new Pair<>(worker.id, alignId_);
        if (cstCache_.containsKey(key))
            return cstCache_.get(key);
        List<BoolExpr> cstList = new ArrayList<>();
        String colVar = "V_COL" + worker.id;
        String rowVar = "V_ROW" + worker.id;
        String groupVar = "V_GROUP" + worker.id;
        String headVar = "V_HEAD" + worker.id;
        String contentVar = "V_CONTENT" + worker.id;
        assert worker.children.size() == 1;
        Node lastChild = worker.children.get(0);
        String childColVar = "V_COL" + lastChild.id;
        BoolExpr eqColCst = z3_.genEqCst(colVar, childColVar);
        String childRowVar = "V_ROW" + lastChild.id;
        BoolExpr eqRowCst = z3_.genEqCst(rowVar, childRowVar);

        String childGroupVar = "V_GROUP" + lastChild.id;
        BoolExpr eqGroupCst = z3_.genEqCst(groupVar, childGroupVar);
        String childHeadVar = "V_HEAD" + lastChild.id;
        BoolExpr eqHeadCst = z3_.genEqCst(headVar, childHeadVar);
        String childContentVar = "V_CONTENT" + lastChild.id;
        BoolExpr eqContentCst = z3_.genEqCst(contentVar, childContentVar);

        cstList.add(eqRowCst);
        cstList.add(eqColCst);
        cstList.add(eqGroupCst);
        cstList.add(eqHeadCst);
        cstList.add(eqContentCst);
        cstCache_.put(key, cstList);
        return cstList;
    }
}
