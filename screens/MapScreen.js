import React from "react";
import MapView, { Marker } from "react-native-maps";
import * as Permissions from "expo-permissions";
import {View, Dimensions, StyleSheet, Text} from "react-native";

export default class MapScreen extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            gasData: null,
            location: {
                latitude: 0,
                longitude: 0,
                latitudeDelta: 0.07,
                longitudeDelta: 0.07,
            },
            markers: null,
            epsilon: 0.00001,
            loading: true
        }
    }

    async componentDidMount() {
        await this.setLocation();
        try  {
            let fuelPrices = await fetch("http://fuelpriceapi.azurewebsites.net/prices/trondelag");
            fuelPrices = await fuelPrices.json();
            if(fuelPrices.length > 0) {
                let markers = this.setMarkers(fuelPrices);
                this.setState(state => ({
                    markers: markers,
                    loading: false
                }));
            }
        } catch(e) {
            console.error(e);
        }
    }

    setMarkers(fuelPrices) {
        // start with one point. If next point is in radius, add to cluster with that point, if not: create new cluster. Check next point if not in the previous clusters, create new cluster etc etc
        let clusters = [];

        if(fuelPrices.length > 0) {
            fuelPrices.forEach(price => {
                if (price.location.includes("undefined")) {
                    fuelPrices.splice(fuelPrices.indexOf(price), 1)
                } else {
                    price.latitude = parseFloat(price.location.split(",")[0].trim());
                    price.longitude = parseFloat(price.location.split(",")[1].trim());
                }
            });
            clusters = [ [fuelPrices.pop()] ];
        } else {
            return;
        }
        while(fuelPrices.length > 0) {
            let point = fuelPrices.pop();
            let avg = [0, 0];
            let assigned = false;

            assigned = clusters.some((cluster) => {
                if(cluster.length === 1) { // Reduce needs two elements
                    avg = [cluster[0].latitude, cluster[0].longitude];
                } else {
                    let averageObject = cluster.reduce((a, b) => {
                        return {
                            latitude: a.latitude + b.latitude,
                            longitude: a.longitude + b.longitude
                        }});
                    avg[0] = averageObject.latitude  / cluster.length;
                    avg[1] = averageObject.longitude  / cluster.length;
                }
                const distance = Math.sqrt((point.latitude - avg[0])**2 + (point.longitude - avg[1])**2);
                if(distance < this.state.epsilon) {
                    cluster.push(point);
                }
                return distance < this.state.epsilon;
            });
            !assigned && clusters.push([point]);
        }

        const today = new Date();
        clusters.forEach(cluster => {
            let clusterObject = {
                gasoline: [],
                diesel: []
            };
            cluster.forEach(point => {
                clusterObject[point.fueltype].push(point);
            });
            clusters[clusters.indexOf(cluster)] = clusterObject;
        });

        clusters.forEach(cluster => {
            for (const key of Object.keys(cluster)) {
                if (cluster[key].length > 0) {
                    cluster[key] = cluster[key].reduce((a, b) => {
                        return (today - new Date(a.Timestamp) < today - new Date(b.Timestamp)) ? a : b;
                    })
                }
            }
        });

        return clusters.map(clusterObject => {
            let hasGasoline = Object.keys(clusterObject["gasoline"]).length !== 0;
            let hasDiesel = Object.keys(clusterObject["diesel"]).length !== 0;

            let idx = "diesel";
            let description = "";
            let title = "";
            if(hasGasoline) {
                idx = "gasoline";
                title = "Gasoline: " + clusterObject["gasoline"]["price"].toString();
                description = clusterObject["gasoline"]["Timestamp"]
            }

            if (hasDiesel) {
                title += "\nDiesel: " + clusterObject.diesel.price.toString();
                description += "\n" + clusterObject.diesel.Timestamp
            }

            clusterObject.key = clusterObject[idx]["RowKey"];
            clusterObject.timestamp = clusterObject[idx]["Timestamp"];
            clusterObject.latitude = clusterObject[idx]["latitude"];
            clusterObject.longitude = clusterObject[idx]["longitude"];
            clusterObject.description = description;
            clusterObject.title = title;

            return clusterObject
        });
    }

    async setLocation() {
        const status = await Permissions.askAsync(Permissions.LOCATION);
        if (status.granted) {
            navigator.geolocation.getCurrentPosition((loc) => {
                this.setState(state => ({
                    location: {
                        ...state.location,
                        latitude: loc.coords.latitude,
                        longitude: loc.coords.longitude
                    }
                }));
            },
                (err) => {
                    console.error(err);
                });
        }
    }

    render() {
        return(
            <View style={styles.container}>
                <MapView
                    style={styles.mapStyle}
                    region={this.state.location}
                >
                    {!this.state.loading && this.state.markers.map(marker => (
                        <Marker
                            coordinate={{latitude: marker.latitude, longitude: marker.longitude}}
                            key={marker.key}
                            title={marker.description}
                        >
                            <View style={{backgroundColor: "lightblue", padding: 1}}>
                                <Text>{marker.title}</Text>
                            </View>
                        </Marker>
                    ))}
                </MapView>
            </View>
        )
    }
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: "#fff",
        alignItems: "center",
        justifyContent: "center",
    },
    mapStyle: {
        width: Dimensions.get("window").width,
        height: Dimensions.get("window").height
    }
});