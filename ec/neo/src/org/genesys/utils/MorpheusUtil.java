package org.genesys.utils;

import krangl.DataCol;
import krangl.DataFrame;
import krangl.IntCol;
import krangl.StringCol;

import javax.xml.crypto.Data;
import java.util.*;

/**
 * Created by yufeng on 9/5/17.
 */
public class MorpheusUtil {

    private static MorpheusUtil instance = null;

    private String prefix_ = "MORPHEUS";

    private int counter_ = 0;

    private List<DataFrame> inputs;

    public static MorpheusUtil getInstance() {
        if (instance == null) {
            instance = new MorpheusUtil();
        }
        return instance;
    }

    public void setInputs(List<DataFrame> ins) {
        inputs = ins;
    }

    public String getMorpheusString() {
        counter_++;
        return prefix_ + counter_;
    }

    private void getSubsets(List<Integer> superSet, int k, int idx, Set<Integer> current, List<Set<Integer>> solution) {
        //successful stop clause
        if (current.size() == k) {
            solution.add(new HashSet<>(current));
            return;
        }
        //unsuccessful stop clause
        if (idx == superSet.size()) return;
        Integer x = superSet.get(idx);
        current.add(x);
        //"guess" x is in the subset
        getSubsets(superSet, k, idx + 1, current, solution);
        current.remove(x);
        //"guess" x is not in the subset
        getSubsets(superSet, k, idx + 1, current, solution);
    }

    public List<Set<Integer>> getSubsets(List<Integer> superSet, int k) {
        List<Set<Integer>> res = new ArrayList<>();
        getSubsets(superSet, k, 0, new HashSet<Integer>(), res);
        return res;
    }

    public List<Set<Integer>> getSubsetsByListSize(int listSize, int k, boolean isPos) {
        List<Integer> data = new ArrayList<>();
        for (int i = 0; i < listSize; i++)
            data.add(i);

        List<Set<Integer>> posSet = getSubsets(data, k);
        if (isPos) return posSet;
        else {
            List<Set<Integer>> negSet = new ArrayList<>();
            for (Set<Integer> pos : posSet) {
                negSet.add(negateSet(pos));
            }
            return negSet;
        }
    }

    public Set<Integer> negateSet(Set<Integer> orgSet) {
        Set<Integer> tgtSet = new HashSet<>();
        for (Integer i : orgSet) {
            if (i == 0) //hack for -0
                tgtSet.add(-99);
            else
                tgtSet.add(i * (-1));
        }
        return tgtSet;
    }

    public void reset() {
        counter_ = 0;
    }

    public Set<String> getHeader(DataFrame df) {
        Set set = new HashSet();
        for (DataCol col : df.getCols()) {
            set.add(col.getName());
        }
        return set;
    }

    public Set<String> getContent(DataFrame df) {
        Set set = new HashSet();
        for (DataCol col : df.getCols()) {
            set.add(col.getName());
        }

        for (List row : df.getRawRows()) {
            for (Object o : row) {
                String val = o.toString();
                if (val.endsWith("0")) {
                    val = removeZeros(val);
                }
                set.add(val);
            }
        }
        return set;
    }

    //compute src - tgt
    public Set setDiff(Set src, Set tgt) {
        Set diff = new HashSet(src);
        diff.removeAll(tgt);
        return diff;
    }

    //Select a sublist from data
    public List<Integer> sel(Set<Integer> selectors, List<Integer> data) {
        assert selectors.size() > 0;
        assert data.size() > 0;
        Set<Integer> negSet = new HashSet<>();
        for (int e : selectors) {
            if (e == -99)
                negSet.add(0);
            else
                negSet.add(-e);
        }
        List<Integer> sublist = new ArrayList<>();
        boolean isNeg = selectors.iterator().next() < 0;
        if (isNeg) {
            for (int j : negSet) {
                if (!data.contains(j)) return new ArrayList<>();
            }
        } else {
            for (int j : selectors) {
                if (!data.contains(j)) return new ArrayList<>();
            }
        }

        for (int i = 0; i < data.size(); i++) {
            if (!isNeg) {
                if (selectors.contains(i))
                    sublist.add(i);
            } else {
                if (!negSet.contains(i))
                    sublist.add(i);
            }
        }
        return sublist;
    }

    // Given a list selector and collist, check whether all selected columns share the same type.
    public boolean hasSameType(List<Integer> sel, List<DataCol> cols) {
        int size = cols.size();
        assert size > 0;
        Set<String> colNames = new HashSet<>();
        for (int idx : sel) {
            assert idx < size : idx;
            String name = cols.get(idx).getClass().getSimpleName();
            if (!name.equals("StringCol")) name = "num";
            colNames.add(name);
        }
        return colNames.size() == 1;
    }

    public String removeZeros(String str) {
        if (!str.contains(".")) return str;
        return str.replaceAll("[0]*$", "").replaceAll(".$", "");

    }

    //Return a list of numeric columns that contains 0.
    public List<String> zeroColumn(DataFrame df) {
        List<String> list = new ArrayList<>();
        for (int i = 0; i < df.getCols().size(); i++) {
            DataCol col = df.getCols().get(i);
            if (col instanceof IntCol) {
                IntCol icol = (IntCol) col;
                for (Integer o : icol.getValues()) {
                    if (o == 0) {
                        list.add(String.valueOf(i));
                        break;
                    }

                }
            }


        }
        return list;
    }

    public int getLen(Object obj) {
        assert obj != null;
        if (obj instanceof Integer)
            return 1;
        else if (obj instanceof List)
            return ((List) obj).size();
        else
            throw new UnsupportedOperationException("Invalid obj." + obj.getClass());
    }

    public int getMax(Object obj) {
        if (obj instanceof Integer)
            return (Integer) obj;
        else if (obj instanceof List) {
            List aList = (List) obj;
            assert !aList.isEmpty();
            int maxIndex = aList.indexOf(Collections.max(aList));
            return (int) aList.get(maxIndex);
        } else {
            throw new UnsupportedOperationException("Invalid obj." + obj.getClass());
        }

    }

    public int getMin(Object obj) {
        if (obj instanceof Integer)
            return (Integer) obj;
        else if (obj instanceof List) {
            List aList = (List) obj;
            assert !aList.isEmpty();
            int minIndex = aList.indexOf(Collections.min(aList));
            return (int) aList.get(minIndex);
        } else {
            throw new UnsupportedOperationException("Invalid obj." + obj.getClass());
        }
    }

    public int getFirst(Object obj) {
        if (obj instanceof Integer)
            return (Integer) obj;
        else if (obj instanceof List) {
            List aList = (List) obj;
            assert !aList.isEmpty();
            return (int) aList.get(0);
        } else {
            throw new UnsupportedOperationException("Invalid obj." + obj.getClass());
        }
    }

    public int getLast(Object obj) {
        if (obj instanceof Integer)
            return (Integer) obj;
        else if (obj instanceof List) {
            List aList = (List) obj;
            assert !aList.isEmpty();
            int e = (int) aList.get(aList.size() - 1);
            return e;
        } else {
            throw new UnsupportedOperationException("Invalid obj." + obj.getClass());
        }
    }

    //Given a collist, compute its HEAD with respect to the inputs
    public int getDiffHead(Set<String> cols) {
        Set<String> set = new HashSet<>(cols);
        for (DataFrame input : inputs) {
            Set headIn = getHeader(input);
            Set contentIn = getContent(input);
            set.removeAll(headIn);
            set.removeAll(contentIn);
        }
        return set.size();
    }

    public static void main(String[] args) {
        MorpheusUtil util_ = MorpheusUtil.getInstance();
        Set<Integer> sel = new HashSet<>(Arrays.asList(1, 3, 6));
//        Set<Integer> sel = new HashSet<>(Arrays.asList(1, 3, 5));
        Set<Integer> sel2 = new HashSet<>(Arrays.asList(-5, -99));
        Set<Integer> sel3 = new HashSet<>(Arrays.asList(-1, -3));
        Set<Integer> sel4 = new HashSet<>(Arrays.asList(0, 2));

        List<Integer> data = Arrays.asList(0, 1, 2, 3, 4, 5);

        System.out.println(MorpheusUtil.getInstance().sel(sel, data));
        System.out.println(MorpheusUtil.getInstance().sel(sel2, data));
        System.out.println(MorpheusUtil.getInstance().sel(sel3, data));
        System.out.println(MorpheusUtil.getInstance().sel(sel4, data));

        String str = "9.00000";
        System.out.println(MorpheusUtil.getInstance().removeZeros(str));
        System.out.println(MorpheusUtil.getInstance().removeZeros("12.0"));
        System.out.println(MorpheusUtil.getInstance().removeZeros("120"));

        List<Integer> aList = Arrays.asList(3, 4, 2, 1, 5);
        List<Integer> aList2 = Arrays.asList(3);
        Object o3 = 9;
        assert util_.getLen(aList) == 5;
        assert util_.getMax(aList) == 5;
        assert util_.getMin(aList) == 1;
        assert util_.getFirst(aList) == 3;
        assert util_.getLast(aList) == 5;

        assert util_.getLen(aList2) == 1;
        assert util_.getMax(aList2) == 3;
        assert util_.getMin(aList2) == 3;
        assert util_.getFirst(aList2) == 3;
        assert util_.getLast(aList2) == 3;

        assert util_.getLen(o3) == 1;
        assert util_.getMax(o3) == 9;
        assert util_.getMin(o3) == 9;
        assert util_.getFirst(o3) == 9;
        assert util_.getLast(o3) == 9;

        System.out.println(MorpheusUtil.getInstance().getSubsetsByListSize(4, 2, true));
        System.out.println(MorpheusUtil.getInstance().getSubsetsByListSize(4, 2, false));
    }

}
