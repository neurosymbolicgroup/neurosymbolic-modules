package org.genesys.interpreter.morpheus;

import krangl.*;
import org.genesys.interpreter.Binop;
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
public class Mutate implements Unop {

    private int lhs;

    private int rhs;

    private Binop binop;

    public Mutate(int l, Binop op, int r) {
        lhs = l;
        binop = op;
        rhs = r;
    }

    public Mutate() {
    }

    public Object apply(Object obj) {
        assert obj instanceof DataFrame;
        DataFrame df = (DataFrame) obj;
        String lhsColName = df.getNames().get(lhs);
        String rhsColName = df.getNames().get(rhs);
        String newColName = MorpheusUtil.getInstance().getMorpheusString();
        String opStr = binop.toString();

        DataFrame res = df.mutate(new TableFormula(newColName, (dataFrame, dataFrame2) -> {
            if (opStr.equals("l(a,b).(/ a b)")) {
                return df.get(lhsColName).div(df.get(rhsColName));
            } else {
                throw new UnsupportedOperationException("Unsupported op:" + opStr);

            }
        }));
        return res;
    }

    public Pair<Boolean, Maybe<Object>> verify(Object obj) {
        List<Pair<Boolean, Maybe<Object>>> args = (List<Pair<Boolean, Maybe<Object>>>) obj;
        Pair<Boolean, Maybe<Object>> arg0 = args.get(0);
        Pair<Boolean, Maybe<Object>> arg1 = args.get(1);
        Pair<Boolean, Maybe<Object>> arg2 = args.get(2);
        Pair<Boolean, Maybe<Object>> arg3 = args.get(3);


        if (!arg0.t1.has()) return new Pair<>(true, new Maybe<>());

        DataFrame df = (DataFrame) arg0.t1.get();
        int lhs = (int) arg2.t1.get();
        Binop op = (Binop) arg1.t1.get();
        int rhs = (int) arg3.t1.get();
        int nCol = df.getNcol();
        if (nCol <= lhs || nCol <= rhs) return new Pair<>(false, new Maybe<>());

        DataCol lhsCol = df.getCols().get(lhs);
        DataCol rhsCol = df.getCols().get(rhs);

        if ((lhsCol instanceof StringCol) || (rhsCol instanceof StringCol)) return new Pair<>(false, new Maybe<>());

        String lhsColName = df.getNames().get(lhs);
        String rhsColName = df.getNames().get(rhs);
        String newColName = MorpheusUtil.getInstance().getMorpheusString();
        String opStr = op.toString();

        DataFrame res = df.mutate(new TableFormula(newColName, (dataFrame, dataFrame2) -> {
            if (opStr.equals("l(a,b).(/ a b)")) {
                System.out.println("Types: " + df.get(lhsColName) + "||||||" + df.get(rhsColName));
                return df.get(lhsColName).div(df.get(rhsColName));
            } else {
                throw new UnsupportedOperationException("Unsupported op:" + opStr);
            }
        }));
//        System.out.println("Mutate:==============" + df.get(lhsColName) + " " + lhsColName + " " + df.get(rhsColName) + " " + rhsColName);
//        Extensions.print(df);
//        Extensions.print(res);
        return new Pair<>(true, new Maybe<>(res));
    }

    public Pair<Object, List<Map<Integer, List<String>>>> verify2(Object obj, Node ast) {
        List<Pair<Object, List<Map<Integer, List<String>>>>> args = (List<Pair<Object, List<Map<Integer, List<String>>>>>) obj;
        Pair<Object, List<Map<Integer, List<String>>>> arg0 = args.get(0);
        Pair<Object, List<Map<Integer, List<String>>>> arg1 = args.get(1);
        Pair<Object, List<Map<Integer, List<String>>>> arg2 = args.get(2);
        Pair<Object, List<Map<Integer, List<String>>>> arg3 = args.get(3);
        List<Map<Integer, List<String>>> conflictList = arg0.t1;

        DataFrame df = (DataFrame) arg0.t0;
//        System.out.println("MUTATE input---------" + df.getCols());
//        Extensions.print(df);
        int lhs = (int) arg2.t0;
        Binop op = (Binop) arg1.t0;
        int rhs = (int) arg3.t0;
        int nCol = df.getNcol();

        Node fstChild = ast.children.get(0);
        Node sndChild = ast.children.get(1);
        Node thdChild = ast.children.get(2);
        Node frdChild = ast.children.get(3);

        if (conflictList.isEmpty())
            conflictList.add(new HashMap<>());


        if (nCol <= lhs || nCol <= rhs || (df.getCols().get(lhs) instanceof StringCol) || (df.getCols().get(rhs) instanceof StringCol) || lhs == rhs) {
            List<String> blackList = new ArrayList<>();
            int maxSize = nCol < 6 ? nCol : 6;
            for (int i = 0; i < maxSize; i++) {
                if (df.getCols().get(i) instanceof StringCol) {
                    blackList.add(String.valueOf(i));
                }
            }
            if (MorpheusGrammar.colMap.get(nCol) != null)
                blackList.addAll(MorpheusGrammar.colMap.get(nCol));

            List<Map<Integer, List<String>>> total = new ArrayList<>();

            if (!blackList.isEmpty()) {
                List<Map<Integer, List<String>>> conflict1 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : conflict1) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    //arg0
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    //arg1
                    partialConflictMap.put(thdChild.id, blackList);
                }
                total.addAll(conflict1);

                List<Map<Integer, List<String>>> conflict2 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : conflict2) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    //arg0
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    //arg1
                    partialConflictMap.put(frdChild.id, blackList);
                }
                total.addAll(conflict2);
            }

            if (MorpheusGrammar.colMap.get(nCol) != null && !MorpheusGrammar.colMap.get(nCol).isEmpty()) {
                List<Map<Integer, List<String>>> bakList2 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : bakList2) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(thdChild.id, MorpheusGrammar.colMap.get(nCol));
                }

                List<Map<Integer, List<String>>> bakList4 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : bakList4) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(frdChild.id, MorpheusGrammar.colMap.get(nCol));
                }

                total.addAll(bakList2);
                total.addAll(bakList4);
            }

            for (int j = 0; j < 6; j++) {
                List<Map<Integer, List<String>>> bakList = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : bakList) {
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
//                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(thdChild.id, Arrays.asList(String.valueOf(j)));
                    partialConflictMap.put(frdChild.id, Arrays.asList(String.valueOf(j)));
                }
                total.addAll(bakList);
            }

            assert !total.isEmpty();
            return new Pair<>(null, total);
        }

        List<String> zeros = MorpheusUtil.getInstance().zeroColumn(df);


        if (!zeros.isEmpty()) {
            for (Map<Integer, List<String>> partialConflictMap : conflictList) {
                //current node.
                partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                //arg0
                partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                partialConflictMap.put(frdChild.id, zeros);
            }
            return new Pair<>(null, conflictList);
        }

        String lhsColName = df.getNames().get(lhs);
        String rhsColName = df.getNames().get(rhs);
        String newColName = MorpheusUtil.getInstance().getMorpheusString();
        String opStr = op.toString();

        DataFrame res = df.mutate(new TableFormula(newColName, (dataFrame, dataFrame2) -> {
            if (opStr.equals("l(a,b).(/ a b)")) {
                return df.get(lhsColName).div(df.get(rhsColName));
            } else {
                throw new UnsupportedOperationException("Unsupported op:" + opStr);
            }
        }));
//        System.out.println("===============return");
//        Extensions.print(res);
        for (Map<Integer, List<String>> partialConflictMap : conflictList) {
            //current node.
            partialConflictMap.put(ast.id, Arrays.asList(ast.function));
            //arg0
            partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
            partialConflictMap.put(sndChild.id, Arrays.asList(sndChild.function));
            partialConflictMap.put(thdChild.id, Arrays.asList(thdChild.function));
            partialConflictMap.put(frdChild.id, Arrays.asList(frdChild.function));
        }
        return new Pair<>(res, conflictList);
    }

    public String toString() {
        return "l(x).(mutate " + " x)";
    }
}
