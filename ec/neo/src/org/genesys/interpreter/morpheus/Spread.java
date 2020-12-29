package org.genesys.interpreter.morpheus;

import krangl.*;
import org.genesys.interpreter.Unop;
import org.genesys.language.MorpheusGrammar;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.Maybe;
import org.genesys.utils.LibUtils;

import java.util.*;

/**
 * Created by yufeng on 9/3/17.
 */
public class Spread implements Unop {

    private int key;

    private int value;

    public Spread(int k, int v) {
        key = k;
        value = v;
    }

    public Spread() {
    }

    public Object apply(Object obj) {
        assert obj instanceof DataFrame;
        DataFrame df = (DataFrame) obj;
//        assert df.getNcol() > key;
//        assert df.getNcol() > value;
        String keyCol = df.getNames().get(key);
        String valCol = df.getNames().get(value);
        DataFrame res = ReshapeKt.spread(df, keyCol, valCol, null, false);
//        System.out.println("----------------SPREAD------------------");
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
        int k = (int) arg1.t1.get();
        int v = (int) arg2.t1.get();
        if ((df.getNcol() <= k) || (df.getNcol() <= v) || (k >= v) || (df.getNrow() == 0)) {
            return new Pair<>(false, new Maybe<>());
        } else {
            String keyCol = df.getNames().get(k);
            String valCol = df.getNames().get(v);
            if (!(df.get(keyCol) instanceof StringCol)) {
                return new Pair<>(false, new Maybe<>());
            }
            Object fstElem = df.getCols().get(k).values$krangl_main()[0];
            if (df.getNames().contains(fstElem)) return new Pair<>(false, new Maybe<>());
            DataFrame res = ReshapeKt.spread(df, keyCol, valCol, null, false);
//            System.out.println("Spread--------------");
//            Extensions.print(res);
            if (res.getNcol() == 0) {
                return new Pair<>(false, new Maybe<>());
            }
            return new Pair<>(true, new Maybe<>(res));
        }
    }

    public Pair<Object, List<Map<Integer, List<String>>>> verify2(Object obj, Node ast) {
        List<Pair<Object, List<Map<Integer, List<String>>>>> args = (List<Pair<Object, List<Map<Integer, List<String>>>>>) obj;
        Pair<Object, List<Map<Integer, List<String>>>> arg0 = args.get(0);
        Pair<Object, List<Map<Integer, List<String>>>> arg1 = args.get(1);
        Pair<Object, List<Map<Integer, List<String>>>> arg2 = args.get(2);
        List<Map<Integer, List<String>>> conflictList = arg0.t1;
        DataFrame df = (DataFrame) arg0.t0;
        int k = (int) arg1.t0;
        int v = (int) arg2.t0;
        int nCol = df.getNcol();

//        System.out.println("Spread--------------" + (df.getNcol() <= k) + " " + (df.getNcol() <= v) + " " + (k >= v));
//        System.out.println("input+++++" + df);

        if (conflictList.isEmpty())
            conflictList.add(new HashMap<>());

        //arg0
        Node fstChild = ast.children.get(0);
        //arg1
        Node sndChild = ast.children.get(1);
        //arg2
        Node thdChild = ast.children.get(2);

        if ((df.getNcol() <= k) || (df.getNcol() <= v) || (k >= v)) {
            List<Map<Integer, List<String>>> conflicts1 = LibUtils.deepClone(conflictList);
            List<Map<Integer, List<String>>> total = new ArrayList<>();

            if (MorpheusGrammar.colMap.get(nCol) != null && !MorpheusGrammar.colMap.get(nCol).isEmpty()) {
                for (Map<Integer, List<String>> partialConflictMap : conflicts1) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(sndChild.id, MorpheusGrammar.colMap.get(nCol));
                }

                List<Map<Integer, List<String>>> conflicts2 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : conflicts2) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(thdChild.id, MorpheusGrammar.colMap.get(nCol));
                }

                total.addAll(conflicts1);
                total.addAll(conflicts2);
            }

            for (int outerIdx = 0; outerIdx < 6; outerIdx++) {
                for (int innerIdx = 0; innerIdx < 6; innerIdx++) {
                    if (outerIdx >= innerIdx) {
                        List<Map<Integer, List<String>>> bakList = new ArrayList<>();
                        Map<Integer, List<String>> eqMap = new HashMap<>();
                        eqMap.put(ast.id, Arrays.asList(ast.function));
//                eqMap.put(fstChild.id, Arrays.asList(fstChild.function));
                        eqMap.put(sndChild.id, Arrays.asList(String.valueOf(outerIdx)));
                        eqMap.put(thdChild.id, Arrays.asList(String.valueOf(innerIdx)));
                        bakList.add(eqMap);
                        total.addAll(bakList);
                    }
                }
            }
            return new Pair<>(null, total);
        } else {
            String keyCol = df.getNames().get(k);
            String valCol = df.getNames().get(v);
            DataFrame res = ReshapeKt.spread(df, keyCol, valCol, null, false);

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
//            Extensions.print(res);
            return new Pair<>(res, conflictList);
        }
    }


    public String toString() {
        return "l(x).(spread " + " x)";
    }
}
