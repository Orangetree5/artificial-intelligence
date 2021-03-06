Noah Mayer's Lernmodell



Einleitung:

Dieser Text setzt voraus das der/die Leser/in schon ein Theoretisches Wissen von Neuronalen Netzwerken
besitzt. Das noetige wissen sollte in einer meiner vorherigen Texte jedoch auch zu finden sein.

Im Feld der Neuronalen Netzwerke existieren eine menge Lernmodelle und das folgende ist eins was mir in den
Sinn fiel. Ob dies ein neues Lernmodell ist, oder nicht, ist mir zur Zeit unbekannt.

Eine Lernmethode die mir bereits bekannt ist, ist die des "Supervised Learning", in anderen Worten: "Ueberwachtes Lernen".
In diesem Prozess werden dem Neuronalen Netzwerk, Inputdaten und die erwarteteten Outputdaten fuer ein Problem gezeigt bis es
die Sorte Problem zu loesen weiss.
Das Modell auf welches ich jedoch kam aehnelt einer anderen Lernmethode ehr bzw. es ist eine Sorte dessen.
Mit dieser Methode lernt das Netzwerk selbststaendig ohne beispielsfaelle, indem es kompetitiv in einer art Wettbewerb gegen sich
selbst ankaempft, bis eine Version von sich, sich als Sieger, und sommit als kompitenteste in dem Bereich, erweisst.



Das Modell:

Gewicht und Bias - Dimension

Der Prozess wuerde dammit beginnen das eine bestimmte Menge an Netzwerken dessen Dimensionen zufallsgemaess generiert wurden, sich der Aufgabe
widmen und ihren Loesungen gemaess einen "Leistungswert" zu geteilt bekommen. Ihre Leistungen wuerden dann verglichen werden: Die Dimensionen
des Netzwerks welches den hoechsten Leistungswert besitzt wuerden sich nicht aendern, doch eine bestimmte Menge an den Netzwerken dessen
Leistungen am niedrigsten waren, wuerden ihre Dimensionen so modifizieren das sie den der erfolgreichsten aehneln. Die uebrigen Netzwerke
wuerden ihre Dimensionen Schrittweise aendern indem sie sich in die Richtung (wenn man die Dimensionen als Raum betrachtet) der anderen
netzwerke begeben (Ihr schritt wuerde natuerlich auch die relevanz der anderen Netzwerke in betracht ziehen). Dann wuerde der komplette 
Prozess wiederholt werden bis sich ein faehiges Netzwerk ergibt.

Die Mathematik:

Definitionen:

N: Ist ein Neuronales Netzwerk; D: Ist ein Vector bestehend aus den Gewichten und Bias', eines Netzwerks; ψ: Ist eine Funktion die,die leistung eines Netzwerks bestimmt; I: Ist eine Funktion die, die relevanz einer Leistung, relativ zu den anderen, darstellt;
η: Ist eine Funktion die steigt wenn der score sinkt; A: Ist die Menge der Neuronalen Netzwerke; Alle Kleinen griechischen
Buchstaben sind Teilmengen der Menge A; t: Ist eine bezeichnung der Runde (jedes mal wenn der Prozess sich wiederholt geht eine
Runde zu ende); δ: Ist eine Konstante, die "Schrittgroesse"; Σ: Ist eine Summe; ΔαD(β) = D(α) - D(β); A minus
einem Griechischen Buchstaben impliziert das Die Menge A ohne der Teilmenge des Griechischen Buchstabens zu betrachten ist.

Das reine Modell:

ΔD(α) = 0; ψ(N(D(α))) > ψ(N(D(A - α)))

ΔD(β) = δ x Σ I(N(D(A - β))) x Δ(A - β)D(β); ψ(N(D(α))) > ψ(N(D(β))) > ψ(N(D(A - α - β)))

D(γ) ≈ D(α); ψ(N(D(A - γ))) > ψ(N(D(γ)))

Submodelle:

Um das Prinzip zu spezialisieren koennten konzepte wie Impuls eingefuehrt werden, in welchem Fall die Formel der Netzwerke β
so aussehen koennten:

ΔD(t + 1)(β) = ΔD(t)(β) + δ x Σ I(N(D(A - β))) x Δ(A - β)D(β); ψ(N(D(α))) > ψ(N(D(β))) > ψ(N(D(A - α - β)))

Man Koennte ebenfalls die Netzwerke ihre eigene Relevanz spueren lassen in dem man ihren ΔD Vectoren mit η multipliziert;
da η ansteigt wenn ψ sinkt wuerde sich das Netzerk schneller von Orten fort bewegen die scheinbar unguenstig sind
und sich langsamer bewegen wenn ihr Gewicht und Bias Vector profitabel wirkt:

ΔD(β) = η x δ x Σ I(N(D(A - β))) x Δ(A - β)D(β); ψ(N(D(α))) > ψ(N(D(β))) > ψ(N(D(A - α - β)))