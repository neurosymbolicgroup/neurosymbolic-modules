package org.genesys.language;

import krangl.DataFrame;
import org.genesys.models.Example;
import org.genesys.models.Problem;
import org.genesys.type.*;
import org.genesys.utils.MorpheusUtil;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by yufeng on 9/6/17.
 */
public class MorpheusGrammar implements Grammar<AbstractType> {

    public AbstractType inputType;

    public AbstractType outputType;

    // maximum column number we need to consider. Can blow up search space
    private int maxCol = 6;

    // max size of column subset. Can blow up search space
    private int maxColListSize = 2;

    private List<Production<AbstractType>> initProductions = new ArrayList<>();

    private List<Production<AbstractType>> inputProductions = new ArrayList<>();

    public static Map<Integer, List<String>> colMap = new HashMap<>();

    public static Map<Integer, List<String>> colListMap = new HashMap<>();

    public static List<String> numList = new ArrayList<>();

    public static List<String> strList = new ArrayList<>();

    public static List<List<Integer>> colListRawData = new ArrayList<>();

    public static List<String> negColList = new ArrayList<>();

    public MorpheusGrammar(Problem p) {
        assert !p.getExamples().isEmpty();
        //FIXME: assume we always only have one example in table domain.
        Example example = p.getExamples().get(0);
        // Rules for inputs
        inputType = new TableType();
        Set constSet = new HashSet();
        for (int i = 0; i < example.getInput().size(); i++) {
            DataFrame input = (DataFrame) example.getInput().get(i);
            //initProductions.add(new Production<>(new TableType(), "input" + i));
            inputProductions.add(new Production<>(new TableType(), "input" + i));
            for (Map<String, Object> row : input.getRows()) {
                constSet.addAll(row.values());
            }
        }

        // Rules for int constants
        for (Object o : constSet) {
            if (o instanceof Number) {
                Production prod = new Production<>(new ConstType(), o.toString());
                prod.setValue(o);
                initProductions.add(prod);
                numList.add(String.valueOf(o));
            } else {
                Production prod = new Production<>(new ConstType(), o.toString());
                prod.setValue(o);
                initProductions.add(prod);
                strList.add(String.valueOf(o));
            }
        }

        // Rule for output.
        this.outputType = new TableType();

        // Rules for column list
        List<Integer> allCols = new ArrayList<>();
        for (int i = 0; i < maxCol; i++) {
            // Rule for possible column name.
            Production prod = new Production<>(new ColIndexType(), i + "");
            prod.setValue(i);
            initProductions.add(prod);
            for (int listSize = 1; listSize < 10; listSize++) {
                List<String> l = new ArrayList<>();
                if (colMap.containsKey(listSize)) {
                    l = colMap.get(listSize);
                } else {
                    colMap.put(listSize, l);
                }
                if (listSize <= i) {
                    l.add(String.valueOf(i));
                }
            }
            allCols.add(i);
        }
        List<Set<Integer>> cols = MorpheusUtil.getInstance().getSubsets(allCols, 1);
        cols.addAll(MorpheusUtil.getInstance().getSubsets(allCols, maxColListSize));
        List<Set<Integer>> negSets = new ArrayList<>();
        for (Set<Integer> col : cols) {
            Set<Integer> negInt = MorpheusUtil.getInstance().negateSet(col);
            negColList.add(negInt.toString());
            negSets.add(negInt);
        }
        //cols.addAll(negSets);
        cols = Stream.concat(negSets.stream(), cols.stream())
                .collect(Collectors.toList());
        for (Set<Integer> ss : cols) {
            colListRawData.add(new ArrayList<>(ss));
            for (int listSize = 1; listSize < 10; listSize++) {
                List<String> l = new ArrayList<>();
                if (colListMap.containsKey(listSize)) {
                    l = colListMap.get(listSize);
                } else {
                    colListMap.put(listSize, l);
                }
                if (ss.size() > listSize) {
                    l.add(ss.toString());
                    continue;
                }

                for (int si : ss) {
                    if (si == -99) continue;
                    if (Math.abs(si) >= listSize) {
                        l.add(ss.toString());
                        break;
                    }
                }
            }
        }
        for (Set<Integer> col : cols) {
            Production prod = new Production<>(new ListType(new IntType()), col.toString());
            prod.setValue(new ArrayList(col));
            initProductions.add(prod);
        }
    }

    @Override
    public AbstractType start() {
        return new InitType(this.outputType);

    }

    @Override
    public String getName() {
        return "MorpheusGrammar";
    }

    @Override
    public List<Production<AbstractType>> getProductions() {
        List<Production<AbstractType>> productions = new ArrayList<>();
        productions.addAll(initProductions);

        productions.add(new Production<>(true, new TableType(), "select", new TableType(), new ListType(new IntType())));
        productions.add(new Production<>(true, new TableType(), "group_by", new TableType(), new ListType(new IntType())));
        productions.add(new Production<>(true, new TableType(), "inner_join", new TableType(), new TableType()));
        productions.add(new Production<>(true, new TableType(), "gather", new TableType(), new ListType(new IntType())));
        productions.add(new Production<>(true, new TableType(), "spread", new TableType(), new ColIndexType(), new ColIndexType()));
        productions.add(new Production<>(true, new TableType(), "unite", new TableType(), new ColIndexType(), new ColIndexType()));
        productions.add(new Production<>(true, new TableType(), "summarise", new TableType(), new AggrType(), new ColIndexType()));
        productions.add(new Production<>(true, new TableType(), "separate", new TableType(), new ColIndexType()));
        productions.add(new Production<>(true, new TableType(), "filter", new TableType(), new BinopBoolType(), new ColIndexType(), new ConstType()));
//        productions.add(new Production<>(true, new TableType(), "filter", new TableType(), new BinopBoolType(), new ColIndexType(), new IntType()));
//        productions.add(new Production<>(true, new TableType(), "filter2", new TableType(), new BinopStringType(), new ColIndexType(), new StringType()));
        productions.add(new Production<>(true, new TableType(), "mutate", new TableType(), new BinopIntType(), new ColIndexType(), new ColIndexType()));

        //FunctionType
        productions.add(new Production<>(new BinopIntType(), "l(a,b).(/ a b)"));

        productions.add(new Production<>(new BinopBoolType(), "l(a,b).(> a b)"));
        productions.add(new Production<>(new BinopBoolType(), "l(a,b).(< a b)"));
        productions.add(new Production<>(new BinopBoolType(), "l(a,b).(== a b)"));
        productions.add(new Production<>(new BinopBoolType(), "l(a,b).(!= a b)"));

        // Aggregator Type
        productions.add(new Production<>(new AggrType(), "mean"));
        productions.add(new Production<>(new AggrType(), "min"));
        productions.add(new Production<>(new AggrType(), "sum"));
        productions.add(new Production<>(new AggrType(), "count"));
        return productions;
    }

    @Override
    public List<Production<AbstractType>> productionsFor(AbstractType symbol) {
        return null;
    }

    @Override
    public AbstractType getOutputType() {
        return this.outputType;
    }

    @Override
    public List<Production<AbstractType>> getInputProductions() {
        return inputProductions;
    }


    @Override
    public List<Production<AbstractType>> getLineProductions(int size) {
        List<Production<AbstractType>> productions = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            productions.add(new Production<>(new TableType(), "line" + i + "table"));
//            productions.add(new Production<>(new BinopIntType(), "line" + i + "binopint"));
//            productions.add(new Production<>(new BinopBoolType(), "line" + i + "binopbool"));
        }

        return productions;
    }

    public List<Production<AbstractType>> getInitProductions() {
        return initProductions;
    }
}
