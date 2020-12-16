package org.genesys.interpreter.morpheus;

import kotlin.jvm.functions.Function2;
import krangl.*;
import org.genesys.interpreter.Unop;
import org.genesys.language.MorpheusGrammar;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.Maybe;
import org.genesys.utils.MorpheusUtil;
import org.genesys.utils.Z3Utils;

import java.util.*;

import static krangl.ColumnsKt.*;
import static krangl.MathHelpersKt.cumSum;
import static krangl.MathHelpersKt.mean;


/**
 * Created by yufeng on 9/3/17.
 */
public class Summarise implements Unop {

    private String aggr;

    private int colVal;

    public Summarise(String a, int c) {
        aggr = a;
        colVal = c;
    }

    public Summarise() {
    }

    private TableFormula getFormula(String colName, String newColName, String aggr) {
        TableFormula tab = new TableFormula(newColName, (df, dataFrame2) -> {
            if (aggr.equals("mean")) {
                return ColumnsKt.mean(df.get(colName), true);
            } else if (aggr.equals("sum")) {
                return sum(df.get(colName), true);
            } else if (aggr.equals("min")) {
                return min(df.get(colName), false);
            } else if (aggr.equals("count")) {
                return count(df.get(colName));
            } else {
                throw new UnsupportedOperationException("Unsupported aggregator:" + aggr);
            }
        });
        return tab;
    }

    public Object apply(Object obj) {
        assert obj instanceof DataFrame;
        DataFrame df = (DataFrame) obj;
        String colName = df.getNames().get(colVal);
        String newColName = MorpheusUtil.getInstance().getMorpheusString();

        DataFrame res = df.summarize(getFormula(colName, newColName, aggr));
        return res;
    }

    public Pair<Boolean, Maybe<Object>> verify(Object obj) {
        List<Pair<Boolean, Maybe<Object>>> args = (List<Pair<Boolean, Maybe<Object>>>) obj;
        Pair<Boolean, Maybe<Object>> arg0 = args.get(0);
        Pair<Boolean, Maybe<Object>> arg1 = args.get(1);
        Pair<Boolean, Maybe<Object>> arg2 = args.get(2);

        if (!arg0.t1.has()) return new Pair<>(true, new Maybe<>());

        DataFrame df = (DataFrame) arg0.t1.get();
        String aggr = (String) arg1.t1.get();
        int colIdx = (int) arg2.t1.get();
        if (df.getNcol() <= colIdx) return new Pair<>(false, new Maybe<>());
//        System.out.println("summarise==================" + df.getCols().get(colIdx));

        if (df.getCols().get(colIdx) instanceof StringCol) return new Pair<>(false, new Maybe<>());

        String colName = df.getNames().get(colIdx);
        String newColName = MorpheusUtil.getInstance().getMorpheusString();


        DataFrame res = df.summarize(getFormula(colName, newColName, aggr));
//        Extensions.print(res);
        return new Pair<>(true, new Maybe<>(res));
    }

    public Pair<Object, List<Map<Integer, List<String>>>> verify2(Object obj, Node ast) {
        List<Pair<Object, List<Map<Integer, List<String>>>>> args = (List<Pair<Object, List<Map<Integer, List<String>>>>>) obj;
        Pair<Object, List<Map<Integer, List<String>>>> arg0 = args.get(0);
        Pair<Object, List<Map<Integer, List<String>>>> arg1 = args.get(1);
        Pair<Object, List<Map<Integer, List<String>>>> arg2 = args.get(2);
        List<Map<Integer, List<String>>> conflictList = arg0.t1;
        DataFrame df = (DataFrame) arg0.t0;
        int nCol = df.getNcol();
        String aggr = (String) arg1.t0;
        int colIdx = (int) arg2.t0;
        Node fstChild = ast.children.get(0);
        Node sndChild = ast.children.get(1);
        Node thdChild = ast.children.get(2);

        if (conflictList.isEmpty())
            conflictList.add(new HashMap<>());

        List<String> strList = new ArrayList<>();
        for (int i = 0; i < nCol; i++) {
            if (df.getCols().get(i) instanceof StringCol) {
                strList.add(String.valueOf(i));
            }
        }


        if (nCol <= colIdx && MorpheusGrammar.colListMap.get(nCol) != null) {
            for (Map<Integer, List<String>> partialConflictMap : conflictList) {
                //current node.
                partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                //arg0
                partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                //arg1
                partialConflictMap.put(thdChild.id, MorpheusGrammar.colListMap.get(nCol));
            }
            return new Pair<>(null, conflictList);
        }

        if ((df.getCols().get(colIdx) instanceof StringCol) && !"count".equals(aggr)) {
            if (!strList.isEmpty()) {
                for (Map<Integer, List<String>> partialConflictMap : conflictList) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    //arg0
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    //arg1
                    partialConflictMap.put(thdChild.id, strList);

                    partialConflictMap.put(sndChild.id, Arrays.asList("min", "mean", "sum"));
                }
            }
            return new Pair<>(null, conflictList);
        }

        String colName = df.getNames().get(colIdx);
        String newColName = MorpheusUtil.getInstance().getMorpheusString();

        DataFrame res = df.summarize(getFormula(colName, newColName, aggr));
        for (Map<Integer, List<String>> partialConflictMap : conflictList) {
            //current node.
            partialConflictMap.put(ast.id, Arrays.asList(ast.function));
            //arg0
            partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
            //arg1
            partialConflictMap.put(sndChild.id, Arrays.asList(sndChild.function));
            //arg2
            partialConflictMap.put(thdChild.id, Arrays.asList(thdChild.function));
        }

        Set<String> eqClasses = new HashSet<>();
        int inSize = df.getNcol() < 6 ? df.getNcol() : 6;
        for (int i = 0; i < inSize; i++) {
            eqClasses.add(String.valueOf(i));
        }
        assert !eqClasses.isEmpty();
        Z3Utils.getInstance().updateEqClassesInPE("HEAD", eqClasses);

        return new Pair<>(res, conflictList);

    }

    public String toString() {
        return "l(x).(summarise " + " x)";
    }
}
