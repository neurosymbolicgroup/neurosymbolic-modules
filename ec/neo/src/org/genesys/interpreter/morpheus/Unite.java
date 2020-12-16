package org.genesys.interpreter.morpheus;

import krangl.DataFrame;
import krangl.Extensions;
import krangl.ReshapeKt;
import org.genesys.interpreter.Unop;
import org.genesys.language.MorpheusGrammar;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.Maybe;
import org.genesys.utils.LibUtils;
import org.genesys.utils.MorpheusUtil;

import java.util.*;

/**
 * Created by yufeng on 9/3/17.
 */
public class Unite implements Unop {

    private String sep_ = "_";

    private boolean remove = true;

    private int lhs;

    private int rhs;

    public Unite(int l, int r) {
        lhs = l;
        rhs = r;
    }

    public Unite() {
    }

    public Object apply(Object obj) {
        assert obj instanceof DataFrame;
        DataFrame df = (DataFrame) obj;
//        assert df.getNames().size() > lhs;
//        assert df.getNames().size() > rhs;
        String lhsCol = df.getNames().get(lhs);
        String rhsCol = df.getNames().get(rhs);
        List<String> colList = new ArrayList<>();
        colList.add(lhsCol);
        colList.add(rhsCol);
        String colName = MorpheusUtil.getInstance().getMorpheusString();
        DataFrame res = ReshapeKt.unite(df, colName, colList, sep_, remove);
//        System.out.println("----------------UNITE------------------");
//        Extensions.print(df);
//        Extensions.print(res);
        return res;
    }

    public Pair<Boolean, Maybe<Object>> verify(Object obj) {
        List<Pair<Boolean, Maybe<Object>>> args = (List<Pair<Boolean, Maybe<Object>>>) obj;
        Pair<Boolean, Maybe<Object>> arg0 = args.get(0);
        Pair<Boolean, Maybe<Object>> arg1 = args.get(1);
        Pair<Boolean, Maybe<Object>> arg2 = args.get(2);

        if (!arg0.t1.has()) return new Pair<>(true, new Maybe<>());

        DataFrame df = (DataFrame) arg0.t1.get();
        int lhs = (int) arg1.t1.get();
        int rhs = (int) arg2.t1.get();
        int nCol = df.getNcol();
        if ((nCol <= lhs) || (nCol <= rhs) || (lhs == rhs)) {
            return new Pair<>(false, new Maybe<>());
        }
        String lhsCol = df.getNames().get(lhs);
        String rhsCol = df.getNames().get(rhs);
        List<String> colList = new ArrayList<>();
        colList.add(lhsCol);
        colList.add(rhsCol);
        String colName = MorpheusUtil.getInstance().getMorpheusString();
        DataFrame res = ReshapeKt.unite(df, colName, colList, sep_, remove);
//        System.out.println("Unite==================");
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
        int lhs = (int) arg1.t0;
        int rhs = (int) arg2.t0;
        int nCol = df.getNcol();

//        System.out.println("Unite==================");
//        System.out.println("input+++++" + df);

        if (conflictList.isEmpty())
            conflictList.add(new HashMap<>());

        //arg0
        Node fstChild = ast.children.get(0);
        //arg1
        Node sndChild = ast.children.get(1);
        //arg2
        Node thdChild = ast.children.get(2);

        if ((nCol <= lhs) || (nCol <= rhs) || (lhs == rhs)) {
            List<Map<Integer, List<String>>> bakList2 = LibUtils.deepClone(conflictList);
            List<Map<Integer, List<String>>> total = new ArrayList<>();

            if (MorpheusGrammar.colMap.get(nCol) != null && !MorpheusGrammar.colMap.get(nCol).isEmpty()) {
                for (Map<Integer, List<String>> partialConflictMap : bakList2) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(sndChild.id, MorpheusGrammar.colMap.get(nCol));
                }

                List<Map<Integer, List<String>>> bakList4 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : bakList4) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(thdChild.id, MorpheusGrammar.colMap.get(nCol));
                }

                total.addAll(bakList2);
                total.addAll(bakList4);
            }

            for (int j = 0; j < 6; j++) {
                List<Map<Integer, List<String>>> bakList = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : bakList) {
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
//                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(sndChild.id, Arrays.asList(String.valueOf(j)));
                    partialConflictMap.put(thdChild.id, Arrays.asList(String.valueOf(j)));
                }
                total.addAll(bakList);
            }
            return new Pair<>(null, total);
        }

        String lhsCol = df.getNames().get(lhs);
        String rhsCol = df.getNames().get(rhs);
        List<String> colList = new ArrayList<>();
        colList.add(lhsCol);
        colList.add(rhsCol);
        String colName = MorpheusUtil.getInstance().getMorpheusString();
        DataFrame res = ReshapeKt.unite(df, colName, colList, sep_, remove);
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
//        Extensions.print(res);
        return new Pair<>(res, conflictList);

    }

    public String toString() {
        return "l(x).(unite " + " x)";
    }
}
