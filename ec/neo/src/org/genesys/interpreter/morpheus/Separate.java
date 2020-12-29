package org.genesys.interpreter.morpheus;

import krangl.DataFrame;
import krangl.Extensions;
import krangl.ReshapeKt;
import krangl.StringCol;
import org.genesys.interpreter.Unop;
import org.genesys.language.MorpheusGrammar;
import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.Maybe;
import org.genesys.utils.MorpheusUtil;

import java.util.*;

/**
 * Created by yufeng on 9/3/17.
 */
public class Separate implements Unop {

    private final String sep_ = "\\.|-|_|\\|";

    private final boolean remove_ = true;

    private final boolean convert_ = false;

    private int colVal;

    public Separate(int v) {
        colVal = v;
    }

    public Separate() {
    }

    public Object apply(Object obj) {
        assert obj instanceof DataFrame;
        DataFrame df = (DataFrame) obj;
        List<String> colArgs = new ArrayList<>();
        String col1 = MorpheusUtil.getInstance().getMorpheusString();
        String col2 = MorpheusUtil.getInstance().getMorpheusString();
        String orgCol = df.getNames().get(colVal);
        colArgs.add(col1);
        colArgs.add(col2);

        DataFrame res = ReshapeKt.separate(df, orgCol, colArgs, sep_, remove_, convert_);
        return res;
    }

    public Pair<Boolean, Maybe<Object>> verify(Object obj) {
        List<Pair<Boolean, Maybe<Object>>> args = (List<Pair<Boolean, Maybe<Object>>>) obj;
        Pair<Boolean, Maybe<Object>> arg0 = args.get(0);
        Pair<Boolean, Maybe<Object>> arg1 = args.get(1);

        if (!arg0.t1.has()) return new Pair<>(true, new Maybe<>());

        DataFrame df = (DataFrame) arg0.t1.get();
        int colIdx = (int) arg1.t1.get();
        if ((df.getNcol() <= colIdx) || (df.getNrow() == 0)) return new Pair<>(false, new Maybe<>());
        if (!(df.getCols().get(colIdx) instanceof StringCol)) return new Pair<>(false, new Maybe<>());
        List<String> colArgs = new ArrayList<>();
        String col1 = MorpheusUtil.getInstance().getMorpheusString();
        String col2 = MorpheusUtil.getInstance().getMorpheusString();
        String orgCol = df.getNames().get(colIdx);
        colArgs.add(col1);
        colArgs.add(col2);
        DataFrame res = ReshapeKt.separate(df, orgCol, colArgs, sep_, remove_, convert_);
//        System.out.println("Running separate...." + orgCol);
//        System.out.println(df);
//        System.out.println(res);

        return new Pair<>(true, new Maybe<>(res));
    }

    public Pair<Object, List<Map<Integer, List<String>>>> verify2(Object obj, Node ast) {
        List<Pair<Object, List<Map<Integer, List<String>>>>> args = (List<Pair<Object, List<Map<Integer, List<String>>>>>) obj;
        Pair<Object, List<Map<Integer, List<String>>>> arg0 = args.get(0);
        Pair<Object, List<Map<Integer, List<String>>>> arg1 = args.get(1);
        List<Map<Integer, List<String>>> conflictList = arg0.t1;

        DataFrame df = (DataFrame) arg0.t0;
        int colIdx = (int) arg1.t0;
        int nCol = df.getNcol();
        Node fstChild = ast.children.get(0);
        Node sndChild = ast.children.get(1);

        if (conflictList.isEmpty())
            conflictList.add(new HashMap<>());

        List<String> noStrList = new ArrayList<>();
        int maxSize = nCol < 6 ? nCol : 6;
        for (int i = 0; i < maxSize; i++) {
            if (!(df.getCols().get(i) instanceof StringCol)) {
                noStrList.add(String.valueOf(i));
            }
        }

        if ((nCol <= colIdx) || !(df.getCols().get(colIdx) instanceof StringCol)) {
            List<String> blackList = new ArrayList<>();
            if (MorpheusGrammar.colMap.get(nCol) != null)
                blackList.addAll(MorpheusGrammar.colMap.get(nCol));
            blackList.addAll(noStrList);
            if (!blackList.isEmpty()) {
                for (Map<Integer, List<String>> partialConflictMap : conflictList) {
                    //current node.
                    partialConflictMap.put(ast.id, Arrays.asList(ast.function));
                    //arg0
                    partialConflictMap.put(fstChild.id, Arrays.asList(fstChild.function));
                    //arg1
                    partialConflictMap.put(sndChild.id, blackList);
                }
            }
            return new Pair<>(null, conflictList);
        }
        List<String> colArgs = new ArrayList<>();
        String col1 = MorpheusUtil.getInstance().getMorpheusString();
        String col2 = MorpheusUtil.getInstance().getMorpheusString();
        String orgCol = df.getNames().get(colIdx);
        colArgs.add(col1);
        colArgs.add(col2);

//        System.out.println("separate===================" + orgCol);
//        Extensions.print(df);
        DataFrame res = ReshapeKt.separate(df, orgCol, colArgs, sep_, remove_, convert_);

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

    public String toString() {
        return "l(x).(separate " + " x)";
    }
}
