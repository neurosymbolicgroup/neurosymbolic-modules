package org.genesys.interpreter.morpheus;

import krangl.*;
import org.genesys.interpreter.Binop;
import org.genesys.interpreter.Unop;
import org.genesys.language.MorpheusGrammar;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.synthesis.MorpheusSynthesizer;
import org.genesys.type.Maybe;
import org.genesys.utils.LibUtils;
import org.genesys.utils.Z3Utils;

import java.util.*;

import static krangl.ColumnsKt.count;
import static krangl.ColumnsKt.min;
import static krangl.ColumnsKt.sum;

/**
 * Created by yufeng on 9/3/17.
 */
public class Filter implements Unop {

    private Binop binop;

    private int lhs;

    private Object rhs;

    private static Map<String, Set<String>> eqCache_ = new HashMap<>();

    public Filter(Binop bin, int l, Object r) {
        binop = bin;
        lhs = l;
        rhs = r;
    }

    public Filter() {

    }

    public Object apply(Object obj) {
        assert obj instanceof DataFrame;
        String op = binop.toString();
        DataFrame df = (DataFrame) obj;
        String colName = df.getNames().get(lhs);

        DataFrame res = df.filter((df1, df2) -> {
            if (op.equals("l(a,b).(> a b)")) {
                return ColumnsKt.gt(df.get(colName), (int) rhs);
            } else if (op.equals("l(a,b).(< a b)")) {
                return ColumnsKt.lt(df.get(colName), (int) rhs);
            } else if (op.equals("l(a,b).(== a b)")) {
                return ColumnsKt.eq(df.get(colName), rhs);
            } else if (op.equals("l(a,b).(!= a b)")) {
                return ColumnsKt.neq(df.get(colName), rhs);
            } else {
                throw new UnsupportedOperationException("Unsupported operator:" + op);
            }
        });
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
        Binop op = (Binop) arg1.t1.get();
        int lhs = (int) arg2.t1.get();
        Object rhs = arg3.t1.get();
        if (df.getNcol() <= lhs) return new Pair<>(false, new Maybe<>());
        if ((df.getCols().get(lhs) instanceof StringCol) && !(rhs instanceof String))
            return new Pair<>(false, new Maybe<>());

        String colName = df.getNames().get(lhs);
        String opStr = op.toString();
        if (opStr.equals("l(a,b).(> a b)")) {
            if (rhs instanceof String) return new Pair<>(false, new Maybe<>());
        } else if (opStr.equals("l(a,b).(< a b)")) {
            if (rhs instanceof String) return new Pair<>(false, new Maybe<>());
        } else if (opStr.equals("l(a,b).(!= a b)")) {
            if (!(rhs instanceof String)) return new Pair<>(false, new Maybe<>());
        }

        DataFrame res = df.filter((df1, df2) -> {
            if (opStr.equals("l(a,b).(> a b)")) {
                return ColumnsKt.gt(df.get(colName), (Number) rhs);
            } else if (opStr.equals("l(a,b).(< a b)")) {
                return ColumnsKt.lt(df.get(colName), (Number) rhs);
            } else if (opStr.equals("l(a,b).(== a b)")) {
                return ColumnsKt.eq(df.get(colName), rhs);
            } else if (opStr.equals("l(a,b).(!= a b)")) {
                return ColumnsKt.neq(df.get(colName), rhs);
            } else {
                throw new UnsupportedOperationException("Unsupported OP:" + opStr);
            }
        });
//        System.out.println("Filter--------------" + colName);
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
        Binop op = (Binop) arg1.t0;
        int lhs = (int) arg2.t0;
        Object rhs = arg3.t0;
        int nCol = df.getNcol();

        Node fstChild = ast.children.get(0);
        Node sndChild = ast.children.get(1);
        Node thdChild = ast.children.get(2);
        Node frdChild = ast.children.get(3);

        if (conflictList.isEmpty())
            conflictList.add(new HashMap<>());

        String opStr = op.toString();
//        System.out.println("==============DF header:" + df.getCols());
//        Extensions.print(df);
        if (df.getNcol() <= lhs || ((df.getCols().get(lhs) instanceof StringCol) && !(rhs instanceof String))
                || (opStr.equals("l(a,b).(> a b)") && (rhs instanceof String)) || ((rhs instanceof String) && opStr.equals("l(a,b).(< a b)"))) {
            List<Map<Integer, List<String>>> all = new ArrayList<>();
            if (MorpheusGrammar.colMap.get(nCol) != null && !MorpheusGrammar.colMap.get(nCol).isEmpty()) {
                // no out of bound access
                List<Map<Integer, List<String>>> conflicts1 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : conflicts1) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(thdChild.id, MorpheusGrammar.colMap.get(nCol));
                }
                all.addAll(conflicts1);
            }

            // > < can't work for string
            if (!MorpheusGrammar.strList.isEmpty()) {
                List<Map<Integer, List<String>>> conflicts2 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : conflicts2) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(sndChild.id, Arrays.asList("l(a,b).(> a b)", "l(a,b).(< a b)"));
                    partialConflictMap.put(frdChild.id, MorpheusGrammar.strList);
                }
                all.addAll(conflicts2);
            }

            // same type
            List<String> strList = new ArrayList<>();
            List<String> noStrList = new ArrayList<>();
            for (int i = 0; i < nCol; i++) {
                if (df.getCols().get(i) instanceof StringCol) {
                    strList.add(String.valueOf(i));
                } else {
                    noStrList.add(String.valueOf(i));
                }
            }

            if (!noStrList.isEmpty() && !MorpheusGrammar.strList.isEmpty()) {
                List<Map<Integer, List<String>>> conflicts3 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : conflicts3) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(thdChild.id, noStrList);
                    partialConflictMap.put(frdChild.id, MorpheusGrammar.strList);
                }
                all.addAll(conflicts3);
            }

            if (!strList.isEmpty() && !MorpheusGrammar.numList.isEmpty()) {
                List<Map<Integer, List<String>>> conflicts4 = LibUtils.deepClone(conflictList);
                for (Map<Integer, List<String>> partialConflictMap : conflicts4) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    partialConflictMap.put(thdChild.id, strList);
                    partialConflictMap.put(frdChild.id, MorpheusGrammar.numList);
                }
                all.addAll(conflicts4);
            }
            return new Pair<>(null, all);
        }

        String colName = df.getNames().get(lhs);
        DataFrame res = df.filter((df1, df2) -> {
            if (opStr.equals("l(a,b).(> a b)")) {
                return ColumnsKt.gt(df.get(colName), (Number) rhs);
            } else if (opStr.equals("l(a,b).(< a b)")) {
                return ColumnsKt.lt(df.get(colName), (Number) rhs);
            } else if (opStr.equals("l(a,b).(== a b)")) {
                return ColumnsKt.eq(df.get(colName), rhs);
            } else if (opStr.equals("l(a,b).(!= a b)")) {
                return ColumnsKt.neq(df.get(colName), rhs);
            } else {
                throw new UnsupportedOperationException("Unsupported OP:" + opStr);
            }
        });


        Set<String> eqClasses = new HashSet<>();
        if(!eqCache_.containsKey(ast.toString())) {
            for (String str : MorpheusGrammar.numList) {
                Double num = Double.valueOf(str);
                Number val;
                if (num.doubleValue() % 1 == 0)
                    val = num.intValue();
                else {
                    val = num.doubleValue();
                }
                if (str.equals(rhs.toString())) {
                    eqClasses.add(str);
                    continue;
                }
                DataFrame eqRes = df.filter((df1, df2) -> {
                    if (opStr.equals("l(a,b).(> a b)")) {
                        return ColumnsKt.gt(df.get(colName), val);
                    } else if (opStr.equals("l(a,b).(< a b)")) {
                        return ColumnsKt.lt(df.get(colName), val);
                    } else if (opStr.equals("l(a,b).(== a b)")) {
                        return ColumnsKt.eq(df.get(colName), val);
                    } else if (opStr.equals("l(a,b).(!= a b)")) {
                        return ColumnsKt.neq(df.get(colName), val);
                    } else {
                        throw new UnsupportedOperationException("Unsupported OP:" + opStr);
                    }
                });
                if (eqRes.getNrow() == res.getNrow()) {
                    eqClasses.add(str);
                }
            }
            eqCache_.put(ast.toString(), eqClasses);
        } else {
            eqClasses = eqCache_.get(ast.toString());
        }
        Z3Utils.getInstance().updateEqClassesInPE("ROW", eqClasses);
        //Working on learning for filter.
        if ((res.getNrow() == df.getNrow()) && MorpheusSynthesizer.learning_) {

            for (Map<Integer, List<String>> partialConflictMap : conflictList) {
                //current node.
                partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                //arg0
                partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                partialConflictMap.put(sndChild.id, Arrays.asList(sndChild.function));
                partialConflictMap.put(thdChild.id, Arrays.asList(thdChild.function));
                partialConflictMap.put(frdChild.id, new ArrayList<>(eqClasses));
            }
            return new Pair<>(null, conflictList);
        }

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
        return "l(x).(filter " + " x)";
    }
}
