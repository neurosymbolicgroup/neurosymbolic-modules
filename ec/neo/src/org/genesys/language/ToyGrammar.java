package org.genesys.language;

import org.genesys.models.MultivalueMap;
import org.genesys.type.AbstractType;

import java.util.List;

/**
 * Created by yufeng on 5/26/17.
 */
public class ToyGrammar implements Grammar<String> {

    private final String start;

    private String name;

    private List<Production<String>> productions;

    private MultivalueMap<String, Production<String>> productionsBySymbol;

    public ToyGrammar(String start, MultivalueMap<String, Production<String>> prods) {
        this.start = start;
        this.productionsBySymbol = prods;
    }

    public void init() {
        productionsBySymbol = new MultivalueMap<>();

        for (Production<String> prod : productions) {
            String func = prod.function;
            String src = prod.source;
            productionsBySymbol.add(src, prod);
        }
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public List<Production<String>> getInputProductions() { return null; }

    @Override
    public List<Production<String>> getLineProductions(int size) { return null; }

    @Override
    public AbstractType getOutputType() { return null; }

    @Override
    public List<Production<String>> getProductions() {
        return productions;
    }

    @Override
    public String start() {
        return this.start;
    }

    @Override
    public List<Production<String>> productionsFor(String symbol) {
//        System.out.println(this.productionsBySymbol + " " + symbol);
        return this.productionsBySymbol.get(symbol);
    }

}
