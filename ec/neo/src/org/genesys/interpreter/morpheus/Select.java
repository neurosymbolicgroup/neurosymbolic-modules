package org.genesys.interpreter.morpheus;

import kotlin.jvm.functions.Function1;
import krangl.DataFrame;
import krangl.Extensions;
import krangl.ReshapeKt;
import org.genesys.interpreter.Binop;
import org.genesys.interpreter.Unop;
import org.genesys.language.MorpheusGrammar;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.Maybe;
import org.genesys.utils.MorpheusUtil;
import org.genesys.utils.Z3Utils;

import java.util.*;

/**
 * Created by yufeng on 9/3/17.
 */
public class Select implements Unop {

    private MorpheusUtil util_ = MorpheusUtil.getInstance();

    public Object apply(Object obj) {
        assert obj != null;
        List pair = (List) obj;
        assert pair.size() == 2 : pair;
        assert pair.get(0) instanceof DataFrame : pair.get(0).getClass();
        assert pair.get(1) instanceof List;
        DataFrame df = (DataFrame) pair.get(0);
        List cols = (List) pair.get(1);
        List<String> colArgs = new ArrayList<>();
        List<Function1> colNegs = new ArrayList<>();
        boolean hasNeg = false;

        for (Object o : cols) {
            Integer index = (Integer) o;
            int absIdx = index;
            if (index == -99) absIdx = 0;
            String arg = df.getNames().get(Math.abs(absIdx));
            colArgs.add(arg);
            if (index < 0) {
                hasNeg = true;
                colNegs.add(Extensions.unaryMinus(arg));
            }
        }
        assert !colArgs.isEmpty();
        DataFrame res;
        if (hasNeg) {
            Function1[] argNegs = colNegs.toArray(new Function1[colNegs.size()]);
            res = Extensions.select(df, argNegs);
        } else {
            res = df.select(colArgs);
        }

        return res;
    }

    public Pair<Boolean, Maybe<Object>> verify(Object obj) {
        List<Pair<Boolean, Maybe<Object>>> args = (List<Pair<Boolean, Maybe<Object>>>) obj;
        Pair<Boolean, Maybe<Object>> arg0 = args.get(0);
        Pair<Boolean, Maybe<Object>> arg1 = args.get(1);

        if (!arg0.t1.has()) return new Pair<>(true, new Maybe<>());

        DataFrame df = (DataFrame) arg0.t1.get();
        List cols = (List) arg1.t1.get();
        int nCol = df.getNcol();
        if (nCol <= cols.size()) {
            return new Pair<>(false, new Maybe<>());
        } else {
            List<String> colArgs = new ArrayList<>();
            List<Function1> colNegs = new ArrayList<>();
            boolean hasNeg = false;

            for (Object o : cols) {
                Integer index = (Integer) o;
                int absIndx = index;
                if (index == -99) absIndx = 0;

                if (nCol <= Math.abs(absIndx)) return new Pair<>(false, new Maybe<>());
                String arg = df.getNames().get(Math.abs(absIndx));
                if (index < 0) {
                    hasNeg = true;
                    colNegs.add(Extensions.unaryMinus(arg));
                }
                colArgs.add(arg);
            }
            assert !colArgs.isEmpty();
            DataFrame res;
            if (!hasNeg) {
                res = df.select(colArgs);
            } else {
                Function1[] argNegs = colNegs.toArray(new Function1[colNegs.size()]);
                res = Extensions.select(df, argNegs);
            }
//            System.out.println("Running select....");
//            System.out.println(df);
//            System.out.println(res);
            return new Pair<>(true, new Maybe<>(res));
        }
    }

    public Pair<Object, List<Map<Integer, List<String>>>> verify2(Object obj, Node ast) {
        List<Pair<Object, List<Map<Integer, List<String>>>>> args = (List<Pair<Object, List<Map<Integer, List<String>>>>>) obj;
        Pair<Object, List<Map<Integer, List<String>>>> arg0 = args.get(0);
        Pair<Object, List<Map<Integer, List<String>>>> arg1 = args.get(1);
        List<Map<Integer, List<String>>> conflictList = arg0.t1;

        DataFrame df = (DataFrame) arg0.t0;
        List cols = (List) arg1.t0;
        int nCol = df.getNcol();
        Node fstChild = ast.children.get(0);
        Node sndChild = ast.children.get(1);

        if (conflictList.isEmpty())
            conflictList.add(new HashMap<>());

        if (nCol <= cols.size()) {
            for (Map<Integer, List<String>> partialConflictMap : conflictList) {
                //current node.
                partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                //arg0
                partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                //arg1
                partialConflictMap.put(sndChild.id, MorpheusGrammar.colListMap.get(nCol));
            }
            return new Pair<>(null, conflictList);
        } else {
            List<String> colArgs = new ArrayList<>();
            List<Function1> colNegs = new ArrayList<>();
            boolean hasNeg = false;

            for (Object o : cols) {
                Integer index = (Integer) o;
                int absIndx = index;
                if (index == -99) absIndx = 0;

                if (nCol <= Math.abs(absIndx)) {
                    for (Map<Integer, List<String>> partialConflictMap : conflictList) {
                        //current node.
                        partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                        //arg0
                        partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                        //arg1
                        partialConflictMap.put(sndChild.id, MorpheusGrammar.colListMap.get(nCol));
                    }
                    return new Pair<>(null, conflictList);
                }
                String arg = df.getNames().get(Math.abs(absIndx));
                if (index < 0) {
                    hasNeg = true;
                    colNegs.add(Extensions.unaryMinus(arg));
                }
                colArgs.add(arg);
            }
            assert !colArgs.isEmpty();
            DataFrame res;
            if (!hasNeg) {
                res = df.select(colArgs);
            } else {
                Function1[] argNegs = colNegs.toArray(new Function1[colNegs.size()]);
                res = Extensions.select(df, argNegs);
            }

            int outSize = res.getNcol();
            int inSize = df.getNcol() < 6 ? df.getNcol() : 6;
            Set<String> eqClasses = new HashSet<>();
//            Extensions.print(df);
            // negative:
            if ((df.getNcol() - outSize) == 1) {
                //only consider negative 1
                List<Set<Integer>> neg1List = util_.getSubsetsByListSize(inSize, 1, false);
                for (Set<Integer> s : neg1List)
                    eqClasses.add(s.toString());
            }
            if ((df.getNcol() - outSize) == 2) {
                //only consider negative 2
                List<Set<Integer>> neg2List = util_.getSubsetsByListSize(inSize, 2, false);
                for (Set<Integer> s : neg2List)
                    eqClasses.add(s.toString());

            }

            // positive:
            if (outSize == 1) {
                //size 1 postive
                List<Set<Integer>> pos1List = util_.getSubsetsByListSize(inSize, 1, true);
                for (Set<Integer> s : pos1List)
                    eqClasses.add(s.toString());

            } else if (outSize == 2) {
                //size 2 positive
                List<Set<Integer>> pos2List = util_.getSubsetsByListSize(inSize, 2, true);
                for (Set<Integer> s : pos2List)
                    eqClasses.add(s.toString());
            }
            Z3Utils.getInstance().updateEqClassesInPE("COL", eqClasses);


            //Compute eq_classes for HEAD.
            Set<String> eqClassesHead = new HashSet<>();
            List<Set<Integer>> headList = new ArrayList<>();
            headList.addAll(util_.getSubsetsByListSize(inSize, 1, true));
            headList.addAll(util_.getSubsetsByListSize(inSize, 1, false));
            headList.addAll(util_.getSubsetsByListSize(inSize, 2, true));
            headList.addAll(util_.getSubsetsByListSize(inSize, 2, false));
            List<Integer> data = new ArrayList<>();
            for (int i = 0; i < inSize; i++) data.add(i);
            List<Integer> orgSel = util_.sel(new HashSet<>(cols), data);
            Set<String> orgStrs = new HashSet<>();

            for (int idx : orgSel) {
                orgStrs.add(df.getNames().get(idx));
            }
            int diffOrg = util_.getDiffHead(orgStrs);

//            System.out.println(cols + "===orgDiff:" + diffOrg);
            for (Set<Integer> myList : headList) {
                List<Integer> actual = util_.sel(myList, data);
                Set<String> colStrs = new HashSet<>();
                for (int idx : actual) {
                    colStrs.add(df.getNames().get(idx));
                }
                int diff = util_.getDiffHead(colStrs);
//                System.out.println(myList + " actual: " + actual + " list:" + colStrs + " diff:" + diff);

                if (diff == diffOrg)
                    eqClassesHead.add(myList.toString());

//                System.out.println( cols + "------> " + myList);

            }
            Z3Utils.getInstance().updateEqClassesInPE("HEAD", eqClassesHead);

//            Extensions.print(res);
            for (Map<Integer, List<String>> partialConflictMap : conflictList) {
                //current node.
                partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                //arg0
                partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                //arg1
                partialConflictMap.put(sndChild.id, Arrays.asList(sndChild.function));
            }
            return new Pair<>(res, conflictList);
        }
    }

    public String toString() {
        return "l(x).(select " + " x)";
    }
}
